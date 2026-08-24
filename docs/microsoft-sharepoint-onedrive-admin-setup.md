# Microsoft SharePoint / OneDrive 24/7 Connector — Admin Setup

Tài liệu này dành cho Microsoft 365 / Entra administrator để bật kết nối SharePoint và OneDrive for Business vào QNSC Knowledge Base.

Mục tiêu vận hành:

\`\`\`text
Microsoft Graph webhook (wakeup)
        ↓
durable notification inbox + dedupe
        ↓
coalesced sync queue / retry / recovery
        ↓
Graph delta cursor (nguồn sự thật)
        ↓
content + ACL + department routing
        ↓
Pending Draft / governance review / publish
        ↓
full reconciliation định kỳ để bắt missed notification, move, delete và cursor hết hạn
\`\`\`

Webhook chỉ là tín hiệu đánh thức. Hệ thống không dùng payload webhook để tự suy đoán toàn bộ trạng thái file; mỗi lần xử lý sẽ đọc Graph delta và lưu cursor bền vững. Vì vậy việc nhận trùng webhook là an toàn và mất một webhook vẫn được bù bằng polling/reconciliation.

## 1. Điều kiện hạ tầng bắt buộc

- Một Microsoft 365 tenant cố định và quyền **Application Administrator** hoặc tương đương để tạo App Registration.
- QNSC API chạy trên URL HTTPS public, có DNS và TLS hợp lệ.
- Microsoft Graph có thể gọi được các endpoint sau từ Internet:
  - \`POST https://<qnskb-host>/api/v1/connectors/webhooks/sharepoint\`
  - \`POST https://<qnskb-host>/api/v1/connectors/webhooks/sharepoint/lifecycle\`
- Redis và worker phải chạy liên tục nếu \`JOB_MODE=celery\`. Nếu chạy \`JOB_MODE=inline\`, API process phải có supervisor/orchestrator tự restart.
- Database migration phải chạy thành công trước khi tạo connector.

Không đặt path webhook vào \`CONNECTOR_WEBHOOK_BASE_URL\`. Biến này chỉ là origin, ví dụ \`https://kb.example.com\`; backend tự nối path.

## 2. Tạo Microsoft Entra App Registration

1. Mở **Microsoft Entra admin center → Identity → Applications → App registrations → New registration**.
2. Đặt tên, ví dụ \`QNSC KB SharePoint OneDrive Connector\`.
3. Chọn **Accounts in this organizational directory only (Single tenant)**.
4. Redirect URI có thể để trống khi dùng app-only. Nếu vẫn cần delegated OAuth cho môi trường dev, thêm Web redirect URI:
   - \`https://<qnskb-host>/api/v1/connectors/oauth/callback\`
5. Lưu lại:
   - **Application (client) ID** → \`MICROSOFT_CLIENT_ID\`
   - **Directory (tenant) ID** → \`MICROSOFT_TENANT_ID\`
6. Vào **Certificates & secrets → New client secret**, đặt thời hạn theo chính sách bảo mật của tenant. Lưu secret value ngay khi tạo; không đưa secret vào connector config, Git hoặc frontend.

Khuyến nghị production: dùng certificate credential thay cho client secret khi quy trình secret rotation của tổ chức đã sẵn sàng.

## 3. Cấp Microsoft Graph permissions

Vào **API permissions → Add a permission → Microsoft Graph → Application permissions**.

### Phương án bring-up nhanh

Cấp và admin-consent:

- \`Sites.Read.All\` — đọc SharePoint sites và libraries.
- \`Files.Read.All\` — đọc file, delta, content và permission snapshot.

Nếu dùng OneDrive theo danh sách user ID, có thể cần \`User.Read.All\` để các quy trình discovery/đối chiếu user của tenant hoạt động. QNSC không tự enumerate toàn tenant khi application mode; nên ưu tiên truyền \`MICROSOFT_ONEDRIVE_USER_IDS\` rõ ràng.

### Phương án least privilege — khuyến nghị production

Dùng \`Sites.Selected\` thay cho \`Sites.Read.All\`, sau đó grant app vào từng site. \`Sites.Selected\` một mình không cấp quyền đọc site nào; phải có bước grant riêng.

Ví dụ gọi Microsoft Graph bằng access token của admin:

\`\`\`http
POST https://graph.microsoft.com/v1.0/sites/{site-id}/permissions
Authorization: Bearer <ADMIN_GRAPH_ACCESS_TOKEN>
Content-Type: application/json

{
  "roles": ["read"],
  "grantedToIdentities": [
    {
      "application": {
        "id": "<QNSC_APP_CLIENT_ID>",
        "displayName": "QNSC KB SharePoint OneDrive Connector"
      }
    }
  ]
}
\`\`\`

Lặp lại cho từng site ID trong \`MICROSOFT_SHAREPOINT_SITE_IDS\`. Nếu tenant dùng quyền selected cho file/library ở mức thấp hơn, cấu hình thêm selected operations tương ứng và test quyền thực tế trước khi production.

Sau khi thêm permissions, bấm **Grant admin consent for <Tenant>** và xác nhận trạng thái là **Granted**.

Tài liệu Microsoft chính thức:

- [Microsoft Graph change notifications overview](https://learn.microsoft.com/en-us/graph/change-notifications-overview)
- [Lifecycle notifications](https://learn.microsoft.com/en-us/graph/change-notifications-lifecycle-events)
- [Subscription resource](https://learn.microsoft.com/en-us/graph/api/resources/subscription?view=graph-rest-1.0)
- [DriveItem delta](https://learn.microsoft.com/en-us/graph/api/driveitem-delta?view=graph-rest-1.0)
- [Grant site permissions](https://learn.microsoft.com/en-us/graph/api/site-post-permissions?view=graph-rest-1.0)
- [Selected permissions overview](https://learn.microsoft.com/en-us/graph/permissions-selected-overview?view=graph-rest-1.0)
- [Microsoft Graph permissions reference](https://learn.microsoft.com/en-us/graph/permissions-reference)
- [Microsoft Graph throttling](https://learn.microsoft.com/en-us/graph/throttling)

## 4. Cấu hình environment của QNSC

Production nên dùng application mode:

\`\`\`dotenv
MICROSOFT_CLIENT_ID="<application-client-id>"
MICROSOFT_CLIENT_SECRET="<client-secret-value>"
MICROSOFT_TENANT_ID="<tenant-guid>"
MICROSOFT_CONNECTOR_AUTH_MODE="application"
MICROSOFT_GRAPH_SCOPE="https://graph.microsoft.com/.default"

# CSV site object IDs đã được grant Sites.Selected.
# Để trống nếu chỉ kết nối OneDrive user.
MICROSOFT_SHAREPOINT_SITE_IDS="<site-object-id-1>,<site-object-id-2>"

# CSV Entra user object IDs có OneDrive for Business cần ingest.
# Để trống nếu chỉ kết nối SharePoint.
MICROSOFT_ONEDRIVE_USER_IDS="<user-object-id-1>,<user-object-id-2>"

# Origin public; KHÔNG thêm /api/v1/... vào cuối.
CONNECTOR_WEBHOOK_BASE_URL="https://kb.example.com"
MICROSOFT_GRAPH_SUBSCRIPTION_MINUTES=1440

# Dispatcher và safety net.
CONNECTOR_SYNC_DISPATCH_INTERVAL_SECONDS=30
CONNECTOR_RECONCILE_INTERVAL_MINUTES=360
CONNECTOR_AUTO_PUBLISH_MODE="governed"
JOB_MODE="celery"
\`\`\`

Không bật \`application\` nếu chưa có client ID, secret, tenant GUID và webhook origin. Không nhập secret vào phần \`config\` khi tạo connector trong UI/API; backend cố ý từ chối các key nhạy cảm ở đó.

\`MICROSOFT_GRAPH_SUBSCRIPTION_MINUTES\` là thời hạn yêu cầu khi tạo/renew subscription. Graph có giới hạn riêng theo resource, vì vậy worker vẫn phải renew thường xuyên và phải có reconciliation để bù subscription hết hạn.

## 5. Deploy và migrate

Chạy migration trong backend:

\`\`\`powershell
cd E:\\QSNC\\QNSC_KB\\qnsc-kb-backend
alembic -c migrations/alembic.ini upgrade head
\`\`\`

Kiểm tra tối thiểu:

\`\`\`powershell
python -m compileall -q src migrations/versions
\`\`\`

Production cần có API, Celery worker và Celery beat. Nếu chỉ chạy API mà đặt \`JOB_MODE=celery\`, webhook sẽ được ghi nhận nhưng không có worker xử lý. Nếu chạy inline, dùng Windows Service/systemd/Kubernetes/Docker restart policy để API tự khởi động lại khi process chết.

## 6. Tạo và khởi tạo connector trong QNSC

1. Đăng nhập tài khoản có quyền \`connector.manage\`.
2. Vào **Admin → Source connectors**.
3. Tạo connector với system **SharePoint**.
4. Application mode không cần bấm OAuth; backend tự lấy client-credentials token từ Entra.
5. Bấm **Discover/Refresh locations**.
6. Chọn các SharePoint libraries/folders hoặc OneDrive drives/folders cần ingest.
7. Chọn department routing mặc định để draft mới được phân loại ngay từ ingestion.
8. Lưu selection.
9. Bấm **Enable webhooks** để tạo Graph subscription cho từng scope.
10. Bấm **Sync now** để chạy first full sync.
11. Kiểm tra trong activity và health badges:
    - queue depth
    - notifications trong 24 giờ
    - reconciliation pending
    - webhook renewal due

Mọi file mới/cập nhật sẽ đi vào pipeline phân loại và \`Pending Draft\` theo governance mặc định. Chỉ khi người có quyền approve/commit thì nội dung mới trở thành bản chính thức. Đây là cấu hình an toàn mặc định để file sai thư mục hoặc file chưa hoàn thiện không tự xuất bản thẳng vào knowledge base.

## 7. Test nghiệm thu bắt buộc

Thực hiện trong một site/folder test trước khi mở rộng:

- Tạo file \`.docx\`, \`.pdf\`, \`.xlsx\` và kiểm tra xuất hiện trong first sync.
- Sửa nội dung file; xác nhận revision mới tạo draft/version mới, không tạo duplicate article.
- Đổi tên và move file; xác nhận external identity vẫn ổn định.
- Xóa file; xác nhận document chuyển \`deleted\` và article/index tương ứng được gỡ hoặc đánh dấu theo governance.
- Gửi lại cùng một Graph notification; xác nhận inbox dedupe và chỉ có một sync request đang active.
- Tạm dừng worker rồi tạo file; bật worker lại; xác nhận durable queue xử lý sau restart.
- Làm hết hạn hoặc revoke subscription; xác nhận lifecycle/renewal health chuyển trạng thái cần chú ý và reconciliation vẫn thu hồi được thay đổi.
- Kiểm tra Graph \`429\`; xác nhận retry/backoff, không tạo hàng nghìn job.
- Chạy full reconciliation sau khi xóa file mà không có webhook; xác nhận stale document được đánh dấu deleted.
- Xác nhận ACL của file vẫn fail-closed: principal chưa map không được mở quyền đọc nội dung nội bộ.

## 8. Vận hành và cảnh báo

Theo dõi endpoint health của connector:

\`\`\`http
GET /api/v1/connectors/{connector_id}/health
\`\`\`

Cần cảnh báo khi:

- \`queue_depth\` tăng liên tục.
- \`last_sync\` quá cũ so với SLA.
- scope có \`full_sync_required=true\` hoặc \`cursor_status != ready\` quá lâu.
- subscription \`reauthorization_required=true\`.
- \`seconds_to_expiry\` nhỏ nhưng renewal không đưa expiry ra xa.
- connector có \`status=error\` hoặc job retry đến terminal failure.
- notifications đến nhưng không có sync job hoàn tất.

Định kỳ rotate client secret/certificate, rà lại site/user allowlist và thu hồi permission khi connector bị gỡ.

## 9. Gỡ quyền / rollback

1. Disable webhook subscriptions trong QNSC hoặc deactivate connector.
2. Xóa app permission grant khỏi từng SharePoint site nếu dùng \`Sites.Selected\`.
3. Revoke client secret/certificate trong Entra.
4. Giữ lại database audit và source metadata theo retention policy; không xóa dữ liệu lịch sử chỉ để “dọn connector”.
5. Nếu cần dừng ingest khẩn cấp, chuyển connector về \`manual\`, giữ worker sống để xử lý queue đang chạy, sau đó disable connector sau khi job đã kết thúc.

