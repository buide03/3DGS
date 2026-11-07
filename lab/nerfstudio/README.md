# Thiết kế lại lab 3DGS với Nerfstudio

Mục tiêu là tái dựng một vật thể từ video, quan sát từ góc mới và đưa vật thể sang cảnh khác. Nerfstudio cung cấp vòng huấn luyện, rasterizer gsplat, Viewer, đánh giá và xuất PLY qua Splatfacto. Phần riêng của lab là tách Gaussian của vật thể và điều khiển pose của nó trong cảnh đích.

## Chạy baseline trên Colab

1. Tải [notebook](3dgs_nerfstudio.ipynb) lên Colab và chọn T4. Notebook dùng `dataset.zip` và `split.json` đã có trong `MyDrive/3dgs_lab`. Đối với clip trụ chắn, đầu ra dự kiến của phần đọc dữ liệu là **78 train, 12 holdout**.
2. Chạy tuần tự đến 500 vòng, đánh giá holdout và xuất PLY. Kết quả nằm dưới `MyDrive/3dgs_lab/nerfstudio`. Lần chạy đầu cần tạo môi trường riêng và biên dịch gsplat, vì vậy thời gian cài đặt tách khỏi thời gian train.
3. Mở PLY trong [SuperSplat](https://superspl.at/editor) để xoay và zoom bằng chuột. Nếu chạy trên máy chủ GPU có SSH, có thể dùng `ns-viewer` với cổng chuyển tiếp để mở Viewer gốc. [Tài liệu Viewer](https://docs.nerf.studio/quickstart/viewer_quickstart.html)
4. Sau khi thấy mô hình, đổi `STEPS` thành `7000` và chạy lại phần train, đánh giá và xuất. Đây là lượt train mới. Giữ cùng tập holdout và ghi độ phân giải thực tế trước khi so sánh với GraphDECO.

## Thay đổi kiến trúc

| Khối | Thiết kế trong Nerfstudio |
| --- | --- |
| Chuẩn bị dữ liệu | Dùng ảnh đã khử méo và các file `sparse/0/*.bin` hiện có. ColmapDataParser đọc intrinsics, pose và điểm thưa. Các file `train_list.txt`, `val_list.txt`, `test_list.txt` giữ đúng tập ảnh từ `split.json`. |
| Train baseline | Splatfacto lấy ảnh đầy đủ, khởi tạo Gaussian từ điểm SfM và dùng gsplat để render khả vi. Tối ưu và điều chỉnh mật độ theo cấu hình Splatfacto. Các mặc định có thể khác GraphDECO dù cùng dùng 3DGS. |
| Đánh giá và xem | Dùng `ns-eval` cho góc holdout và xuất Gaussian PLY bằng `ns-export gaussian-splat`. PLY baseline biểu diễn cả cảnh nhìn thấy trong video. |
| Tách vật thể | Mở rộng Splatfacto bằng một phương thức riêng. Loader giữ mask khớp với ảnh và dùng các mask train để lọc điểm SfM ban đầu. Model tính loss RGB trong foreground và loss silhouette từ `outputs["accumulation"]`. |
| Ghép cảnh | Nạp PLY vật thể và PLY cảnh đích, đưa chúng vào cùng hệ tọa độ, rồi render chung bằng gsplat. Tách pose camera và pose vật thể để kiểm tra góc nhìn mới, chuyển động và che khuất độc lập. |

[Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) · [ColmapDataParser](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/data/dataparsers/colmap_dataparser.py) · [Mã Splatfacto](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/models/splatfacto.py)

## Thiết kế bước tách và di chuyển vật thể

1. **Kiểm tra nhanh bằng crop 3D.** Đặt hộp bao quanh vật thể trong Viewer gốc hoặc dùng các tham số `--obb-center`, `--obb-rotation`, `--obb-scale` của exporter. Xoay PLY đã cắt để kiểm tra nền còn sót. Crop có thể giữ lại Gaussian của nền nằm trong hộp. [Exporter Nerfstudio](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/scripts/exporter.py)
2. **Huấn luyện vật thể với mask.** Chiếu điểm thưa vào các ảnh train và giữ điểm nằm trong foreground ở nhiều góc. Khi train, tính RGB L1 trên pixel foreground và đối chiếu alpha tích lũy của Splatfacto với mask trên toàn ảnh. Cách này dùng trực tiếp alpha của gsplat thay cho pass render trắng riêng trong bản GraphDECO. Báo cáo PSNR foreground và IoU silhouette trên cùng 12 ảnh holdout.
3. **Đóng gói asset.** Lưu Gaussian PLY cùng metadata gồm pivot, trục lên, tỷ lệ và phép biến đổi từ hệ COLMAP sang hệ train. Với video đơn camera, chiều cao có thể chuẩn hóa theo vật thể. Dùng kích thước đã đo nếu cần đặt theo mét.
4. **Ghép và render.** Thay đổi tâm và covariance của Gaussian theo pose vật thể. Với SH bậc cao, đánh giá màu theo hướng nhìn trong hệ của vật thể khi nó xoay. Ghép Gaussian vật thể với cảnh đích trước khi rasterize để kiểm tra che khuất bằng demo cho vật thể đi sau một vật cản.

Notebook hiện triển khai **baseline, đánh giá và xuất PLY**. Các bước tách vật thể có mask và ghép cảnh ở trên là thiết kế cho mốc tiếp theo. Có thể dùng cùng dữ liệu và kết quả GraphDECO để so sánh khi triển khai các mốc đó.
