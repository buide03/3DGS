# Lab 3DGS trên Colab T4

Notebook [3dgs_t4.ipynb](3dgs_t4.ipynb) chạy theo [pipeline đã thống nhất](../doc/ghi_nho_lab_3DGS.md). Nó dùng mã GraphDECO tại commit `54c035f7834b564019656c3e3fcc3646292f727d` và ghi kết quả vào Google Drive.

## Bắt đầu

1. Quay một vật thể đứng yên từ nhiều góc bằng camera sau 1×. Ưu tiên video gốc 1080p/30 FPS nếu có. Dùng vật thể có họa tiết, tránh khung hình mờ và giữ vật thể trong ảnh khi đi vòng quanh.
2. Tạo thư mục `MyDrive/3dgs_lab` trong Drive và đưa video vào đó dưới tên `capture.mov`. Nếu dùng tên khác, sửa biến `VIDEO` trong notebook.
3. Từ thư mục gốc của repo trên Windows, tạo ZIP chứa trực tiếp các script, rồi tải ZIP lên cùng thư mục Drive.

   ```powershell
   Compress-Archive -Path lab\*.py -DestinationPath lab.zip
   ```

4. Mở Colab, tải notebook lên và chọn GPU T4. Chạy các cell cài đặt, COLMAP, train thử 500 vòng và train baseline. Kiểm tra số camera COLMAP đăng ký và ảnh render holdout trước khi chuyển sang bước mask. Notebook tách đều 90 frame theo thời gian, nên chất lượng video và độ phủ góc quay ảnh hưởng trực tiếp đến dữ liệu đầu vào.
5. Tải `mask_inputs.zip` mà notebook tạo. Vẽ mask nhị phân cho vật thể trên từng ảnh đã khử méo. Nền đen, vật thể trắng, ảnh và mask cùng kích thước. Ví dụ `00007.jpg` đi cùng `00007.png`. Nén các PNG trực tiếp vào `masks.zip` và tải lên Drive. Chạy phần train vật thể, đánh giá và demo.

## Xem tương tác sau khi train

Chạy mục **Viewer tương tác sau khi train** trong notebook ngay khi có PLY baseline hoặc PLY vật thể. Chọn mô hình và vòng lặp, chỉnh góc ngang, góc cao hoặc zoom, rồi bấm **Render góc nhìn**. Colab render trên GPU và gửi ảnh về trình duyệt. Khi mở phiên mới, gắn Drive và cài môi trường trước khi mở viewer. Nếu đã tải `lab.zip` lên trước khi có viewer, tạo và tải lại ZIP chứa `viewer.py`, rồi giải nén lại trong một runtime mới.

## Kết quả trên Drive

| Đường dẫn | Nội dung |
| --- | --- |
| `dataset.zip`, `split.json` | Dữ liệu COLMAP và danh sách train/holdout. |
| `baseline/`, `object/` | Cấu hình train, checkpoint, mô hình Gaussian và ảnh render ở góc holdout. |
| `comparison/` | Ảnh thật, baseline và bản tách vật thể đặt cạnh nhau. `scores.json` ghi PSNR trên vùng foreground cho từng góc giữ riêng. |
| `demo/asset.ply`, `demo/asset.json` | Asset theo định dạng GraphDECO và metadata để đặt nó vào hệ tọa độ cảnh đích. |
| `demo/orbit.mp4`, `demo/placement.mp4` | Video camera quay quanh vật thể và video vật thể di chuyển qua sau vật cản. |

## Các quyết định trong bản lab

- COLMAP chạy CPU. Train và render chạy trên GPU. Baseline mặc định 7.000 vòng, ảnh rộng 800 px và densification dừng ở vòng 5.000. Nếu hết VRAM, giảm `RESOLUTION` hoặc `DENSIFY_UNTIL`.
- Bản đầu dùng mask vẽ thủ công. `train_object.py` ghép thêm bước lọc điểm khởi tạo và loss foreground vào trainer đã ghim. Pass phụ render Gaussian trắng trên nền đen để thu alpha tích lũy, nên lượt train với mask tốn thêm thời gian và VRAM.
- Cảnh đích trong `demo.py` gồm sàn và vật cản tạo bằng Gaussian. Asset và cảnh được render chung. Chiều cao vật thể được chuẩn hóa về 1 đơn vị và hướng lên được ước lượng từ camera. Nếu hướng lên bị lệch, truyền `--up X Y Z` cho script demo.
- Khi đổi video, dùng thư mục Drive và thư mục runtime mới. Nếu thay mask trong cùng phiên, xóa bản mask đã giải nén trong runtime trước khi chạy lại cell mask. Dùng biến `RESUME` để tiếp tục từ một checkpoint của đúng lượt train và bộ dữ liệu.
