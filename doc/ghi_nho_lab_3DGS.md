# Ghi nhớ kiến trúc lab 3DGS

Lab nhận video điện thoại quay quanh một vật thể đứng yên. Đầu ra gồm ảnh từ các góc chưa dùng để huấn luyện và một asset Gaussian có thể tịnh tiến, xoay, đổi tỷ lệ trong cảnh 3D khác. Bản đầu dùng một vật thể mờ, có họa tiết, và một cảnh đích đơn giản có mặt sàn cùng vật cản.

## Pipeline đã chốt

Triển khai trong [notebook Colab T4](../lab/3dgs_t4.ipynb). Xem [cách chuẩn bị video, script và mask](../lab/README.md).

**Video điện thoại → frame chọn lọc → COLMAP → 3DGS baseline → 3DGS có mask → asset Gaussian → demo trong cảnh mới.**

| Bước | Chạy ở đâu | Đầu ra và điều kiện chuyển bước |
| --- | --- | --- |
| 1. Chuẩn bị video | Máy cá nhân | Quay quanh một vật thể đứng yên ở vài độ cao, giữ ánh sáng và tiêu cự ổn định. Chọn 60–120 frame rõ, đủ độ chồng lấp. |
| 2. Ước lượng camera và điểm thưa | Colab | Chạy COLMAP trên ảnh đầy đủ. Lưu ảnh đã khử méo, pose, intrinsics, điểm thưa và danh sách train/holdout. Kiểm tra các camera đăng ký được từ nhiều phía và có điểm thưa trên vật thể. Giữ riêng khoảng 10–15% góc rải quanh vật thể để đánh giá. Pose của chúng được ước lượng chung trong SfM, còn ảnh của chúng không tham gia tối ưu Gaussian. |
| 3. Train baseline | Colab T4 | Dùng GraphDECO tại commit đã ghim. Chạy thử 500 vòng để bắt lỗi cài đặt và dữ liệu, sau đó train 7.000 vòng. Lưu PLY, checkpoint và ảnh render tại các góc holdout để so với ảnh thật. |
| 4. Tách vật thể | Tạo và rà mask trên máy cá nhân, train trên Colab T4 | Mask khớp từng ảnh đã khử méo. Khởi tạo từ điểm SfM thuộc vật thể theo nhiều mask train. Train với loss RGB trong mask và ràng buộc silhouette. Dùng cùng bộ holdout để so với baseline, rồi render một vòng camera tự tạo để kiểm tra viền và nền còn sót. |
| 5. Xuất asset và ghép cảnh | Colab T4 | Xuất PLY Gaussian và JSON ghi pivot, hệ trục, tỷ lệ và bậc SH. Đặt asset vào một cảnh Gaussian có sàn và vật cản. Xuất video cho thấy vật thể đổi vị trí, xoay và được che đúng khi đi qua vật cản. Có thể xem video trên máy cá nhân. |

**Cấu hình baseline T4.** Sau khi kiểm tra GPU bằng `nvidia-smi` và build được các extension CUDA, chạy `train.py` với `--eval --iterations 7000 --resolution 800 --data_device cpu --checkpoint_iterations 2000 5000 7000`. Tùy chọn `--eval` bật chia tập mặc định của upstream. Khi dùng danh sách holdout riêng, loader phải dùng đúng danh sách đã lưu. Nếu hết VRAM, giảm độ phân giải trước, rồi cân nhắc giảm `--densify_until_iter`.

**Lưu dữ liệu.** Giữ bộ dữ liệu COLMAP và danh sách train/holdout trên Drive để tái sử dụng giữa các lượt. Train từ bản sao trong ổ đĩa runtime Colab và sao lưu checkpoint, PLY, cấu hình và ảnh đánh giá về Drive trong quá trình chạy. Checkpoint dùng để tiếp tục tối ưu, PLY dùng để render và ghép cảnh.

## 1. Kiến trúc từ mã nguồn của nhóm tác giả

Tham chiếu [GraphDECO/Inria](https://github.com/graphdeco-inria/gaussian-splatting/tree/54c035f7834b564019656c3e3fcc3646292f727d), commit `54c035f`. Bản clone để đọc nằm tại `work/gaussian-splatting` bên cạnh repo `3DGS`. Các submodule CUDA và viewer cần được lấy khi chuẩn bị môi trường chạy. [Bài báo gốc](../Sources/3d_gaussian_splatting_low.pdf)

| Thành phần | Vai trò trong repo gốc | Dùng trong lab |
| --- | --- | --- |
| `convert.py` | Gọi COLMAP để tìm và ghép đặc trưng, ước lượng camera, tạo điểm thưa và khử méo ảnh. | Tách frame từ video trước khi chạy. Giữ ảnh đã khử méo và mô hình COLMAP làm đầu vào chuẩn. |
| `scene/dataset_readers.py`, `scene/cameras.py` | Đọc ảnh, pose, intrinsics và điểm thưa, rồi tạo camera cho renderer. | Gắn mask theo tên frame và chia train/holdout. Dùng cùng phép khử méo và resize cho ảnh, mask và intrinsics. |
| `scene/gaussian_model.py` | Khởi tạo và quản lý tâm, scale, rotation, opacity, SH. Thực hiện clone, split, prune và lưu PLY. | Tái sử dụng model và density control. Khởi tạo từ các điểm thưa thuộc vật thể. |
| `train.py`, `gaussian_renderer/` | Chọn camera, gọi rasterizer CUDA, tính L1 và D-SSIM, lan truyền gradient rồi tối ưu. | Giữ pipeline gốc cho baseline. Sau đó thêm ràng buộc mask để tách vật thể. |
| `render.py` | Nạp checkpoint và render các camera train/test đã nạp. | Thêm đường camera quỹ đạo với pose tự tạo để render từ góc mới. |

### Đối chiếu cấu trúc hai repo

| Nhóm trong GraphDECO | Vị trí hoặc cách dùng trong lab |
| --- | --- |
| `convert.py` | `lab/prepare.py` gọi COLMAP và bổ sung lấy frame từ video. |
| `train.py`, `arguments/`, `scene/` | Notebook gọi trainer gốc cho baseline. `lab/train_object.py` mở rộng bước nạp điểm và tính loss cho mask, dùng lại cấu hình camera, model và density control từ upstream. |
| `gaussian_renderer/`, `submodules/` | Notebook cài các extension tại commit ghim. Train baseline, train vật thể và demo dùng chung rasterizer gốc. |
| `render.py`, `metrics.py`, `full_eval.py` | Lab gọi `render.py` để render holdout và tính PSNR foreground trong notebook. `lab/demo.py` xử lý camera orbit, xuất asset và ghép cảnh. |
| `SIBR_viewers/` | Ứng dụng desktop của upstream. Bản lab dùng hai video MP4 làm đầu ra trình diễn. |

Khi đọc mã, bắt đầu từ `lab/3dgs_t4.ipynb` để thấy thứ tự chạy, rồi đọc `train.py` → `scene/gaussian_model.py` → `gaussian_renderer/__init__.py` trong upstream để lần theo một vòng huấn luyện. Ba script trong `lab/` tập trung vào các phần riêng của bài lab.

## 2. Quay và tái dựng vật thể

1. **Quay.** Giữ vật thể và nền đứng yên, đi vòng quanh với điện thoại ở vài độ cao. Quay đủ các mặt cần tái dựng, giữ tiêu cự và ánh sáng ổn định. Lấy khoảng 60–120 frame rõ, có độ chồng lấp. Đây là mốc thử ban đầu và có thể điều chỉnh theo video.
2. **Chạy baseline.** Đưa frame vào thư mục `input/` của bộ dữ liệu COLMAP. Kiểm tra camera có tạo thành một quỹ đạo liền mạch và đám mây điểm có điểm trên vật thể. Giữ riêng một số góc có ảnh thật để đánh giá. Có thể dùng các ảnh holdout trong bước SfM để lấy pose của chúng, rồi loại chúng khỏi loss và bước cập nhật Gaussian. Ghi rõ giao thức này khi báo cáo kết quả.
3. **Tách vật thể.** Tạo mask 2D cho từng frame và kiểm tra viền, lỗ rỗng và vùng bị che. Dùng ảnh đầy đủ để COLMAP có đủ đặc trưng. Sau SfM, lọc điểm khởi tạo dựa trên sự đồng thuận của mask ở nhiều quan sát. Khi huấn luyện, tính loss RGB trong mask và thêm ràng buộc silhouette giữa alpha tích lũy của renderer và mask. Mask chỉ dùng để bỏ qua pixel ngoài vật thể sẽ bỏ ngỏ các Gaussian nền. Nếu API rasterizer đang dùng chưa trả alpha, cần bổ sung đầu ra alpha khả vi hoặc một pass alpha tương đương trước khi thêm loss silhouette.
4. **Render góc mới.** Đánh giá bằng ảnh holdout tại pose đã ước lượng, sau đó render một vòng camera tự tạo nhìn vào tâm vật thể. Đối chiếu hình dạng, viền, chi tiết mảnh và độ ổn định qua từng góc. Mặt khuất chưa từng được quay thường thiếu dữ liệu để tái dựng đáng tin cậy.

## 3. Đóng gói và di chuyển trong cảnh khác

- **Asset.** Xuất PLY theo schema Gaussian của mã gốc và một JSON ghi hệ trục, pivot, scale, bậc SH và đơn vị. Đặt pivot tại đáy vật thể để dễ đặt lên sàn. SfM từ video một camera có tỷ lệ tuyệt đối chưa xác định. Nếu cần đơn vị mét, dùng một kích thước đã đo để hiệu chuẩn.
- **Pose của asset.** Với tâm cục bộ $\mu$ và hiệp phương sai $\Sigma$, phép quay $R$, tỷ lệ đều $s>0$ và tịnh tiến $t$ cho $\mu'=sR\mu+t$ và $\Sigma'=s^2R\Sigma R^\top$. Nếu lưu scale và quaternion, nhân scale với $s$ và hợp thành quaternion với phép quay của asset. Dùng transform khi render để có thể đổi pose mà vẫn giữ asset gốc.
- **Màu theo hướng nhìn.** Nếu xoay asset mà giữ nguyên hệ số SH, cần đổi hướng nhìn từ hệ tọa độ thế giới về hệ cục bộ trước khi tính SH. Bản demo đầu có thể dùng SH bậc 0 để kiểm tra pose và che khuất trước. Màu học từ video mang theo ảnh hưởng ánh sáng lúc quay, nên đặt vào cảnh mới chưa tự tạo ánh sáng hay bóng đổ phù hợp.
- **Ghép cảnh.** Dùng một cảnh đích cũng biểu diễn bằng Gaussian cho bản đầu. Đưa asset và cảnh về cùng hệ tọa độ, rồi render chung để rasterizer sắp thứ tự theo độ sâu trong từng tile. Điều này cho phép thử vật cản nằm trước hoặc sau asset. Hai ảnh render tách biệt rồi chồng lên nhau theo một thứ tự cố định sẽ sai khi thứ tự che khuất thay đổi theo pixel.

## 4. Kiến trúc demo tham khảo

[Trang dự án của tác giả](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) trình bày video và các kết quả so sánh. Phần xem tương tác được phát hành riêng dưới dạng ứng dụng desktop SIBR. [Hướng dẫn viewer của GraphDECO](https://github.com/graphdeco-inria/gaussian-splatting#interactive-viewers)

| Chế độ của tác giả | Luồng hoạt động |
| --- | --- |
| SIBR Real-Time Viewer | Nạp thư mục mô hình đã train, nhận pose từ bộ điều khiển camera và render Gaussian trên GPU của máy chạy viewer. Người xem có thể di chuyển tự do hoặc chuyển đến một camera đã chụp. |
| SIBR Network Viewer | Kết nối đến tiến trình train, gửi pose camera và nhận ảnh do tiến trình train render. Dùng để quan sát mô hình trong khi tối ưu. Cổng mặc định là 6009. |

**Áp dụng cho lab.** Tách train và demo tại đầu ra PLY. Train trên Colab T4, lưu PLY và cấu hình, rồi dùng một script demo nhận hai đầu vào riêng là pose camera và pose vật thể. Giữ vật thể cố định khi chạy quỹ đạo camera để kiểm tra góc nhìn mới. Sau đó giữ camera cố định, thay đổi pose vật thể và render chung với cảnh đích để kiểm tra che khuất. Có thể bố trí kết quả giống trang tác giả, với ảnh thật cạnh ảnh render ở các góc holdout, một video orbit và một video di chuyển vật thể.

Bản đầu render sẵn trên Colab và xuất MP4 để xem trên máy cá nhân. Nếu máy đáp ứng yêu cầu CUDA và OpenGL của SIBR, dùng Real-Time Viewer để khám phá mô hình baseline sau huấn luyện. Điều khiển pose riêng của asset là phần bổ sung của lab, vì điều hướng camera trong viewer gốc chỉ thay đổi góc nhìn vào cảnh đã train.

## 5. Thứ tự triển khai

| Mốc | Kết quả cần thấy |
| --- | --- |
| A. Baseline video → 3DGS | COLMAP đăng ký được các góc quanh vật thể, huấn luyện tạo PLY và render được ảnh holdout. |
| B. Asset riêng | Orbit cho thấy vật thể trên nền mới, viền giữ được qua nhiều góc và ít kéo theo nền cũ. So với baseline trên cùng tập holdout, có thể dùng PSNR trong mask và IoU silhouette khi có alpha. |
| C. Đặt vào cảnh mới | Cùng một asset xuất hiện ở hai vị trí và hướng khác nhau trên sàn. Một vật cản lần lượt nằm trước và sau asset khi đổi vị trí hoặc góc camera. |

Giữ các script lab trong repo riêng và gọi hoặc mở rộng mã upstream tại commit đã ghim. Môi trường baseline cần COLMAP, PyTorch/CUDA và các extension theo [hướng dẫn upstream](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/README.md). Phạm vi sử dụng mã tham chiếu được mô tả trong [giấy phép upstream](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/LICENSE.md).
