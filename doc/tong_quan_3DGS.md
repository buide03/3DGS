# Tổng quan 3D Gaussian Splatting (3DGS)

## 1. Đầu vào từ SfM

Bắt đầu với một tập ảnh chụp cùng một cảnh từ nhiều góc nhìn. Structure from Motion (SfM) tạo đám mây điểm thưa và ước lượng thông số nội tại, tư thế của các camera. 3DGS dùng các điểm thưa để khởi tạo Gaussian, sau đó dùng ảnh gốc và thông số camera tương ứng để huấn luyện.

## 2. Khởi tạo các Gaussian 3D

- **Vị trí tâm:** Mỗi Gaussian ban đầu có tâm trùng với một điểm SfM. Tâm có thể dịch chuyển trong quá trình tối ưu.
- **Opacity (độ đục):** Mã tham chiếu khởi tạo opacity ở mức 0,1 bằng cách lưu giá trị logit tương ứng. Khi lấy opacity để render, hàm sigmoid chuyển tham số này về khoảng (0, 1).
- **Hiệp phương sai:** Ma trận $\Sigma = RSS^\top R^\top$ xác định hình dạng và hướng của Gaussian. $S$ là ma trận scale theo ba trục và $R$ là ma trận quay suy ra từ quaternion. Scale ban đầu giống nhau trên ba trục và dựa vào khoảng cách đến các điểm lân cận.
- **Màu sắc:** Mỗi Gaussian lưu các hệ số Spherical Harmonics (SH) cho ba kênh RGB. Màu của điểm SfM khởi tạo hệ số bậc 0, còn các hệ số bậc cao bắt đầu từ 0. Khi render, màu được tính từ các hệ số này và hướng nhìn.

## 3. Chiếu và rasterization

1. Dùng thông số camera để chiếu tâm Gaussian lên mặt ảnh và biến đổi hiệp phương sai, thu được một Gaussian 2D có thể phủ nhiều pixel.
2. Chia ảnh thành các tile **16 × 16 pixel**. Đưa mỗi Gaussian vào danh sách của mọi tile mà vùng ảnh hưởng của nó chạm tới. Sắp xếp các mục theo ID tile rồi theo độ sâu, để mỗi tile có danh sách Gaussian từ gần đến xa.
3. GPU xử lý các pixel song song. Mỗi pixel duyệt danh sách của tile và trộn màu từ gần đến xa. Mức đóng góp của mỗi Gaussian phụ thuộc vào opacity, giá trị Gaussian 2D tại pixel và lượng ánh sáng còn truyền qua các Gaussian phía trước.

## 4. Tối ưu bằng ảnh gốc

Mỗi vòng lặp chọn một góc nhìn huấn luyện, render ảnh và so sánh với ảnh gốc tương ứng bằng loss kết hợp L1 và D-SSIM. Lan truyền ngược qua rasterizer khả vi tính gradient. Adam dùng gradient để cập nhật tâm, scale, rotation, opacity và các hệ số SH.

## 5. Điều chỉnh mật độ Gaussian

Adaptive Density Control định kỳ dùng độ lớn gradient của loss theo vị trí tâm đã chiếu lên ảnh để chọn Gaussian cần bổ sung. Khi gradient đủ lớn, Gaussian nhỏ được nhân bản (**clone**), còn Gaussian lớn được thay bằng các Gaussian nhỏ hơn (**split**). Bước **prune** loại các Gaussian có opacity quá thấp và có thể loại cả những Gaussian quá lớn khi áp dụng ngưỡng kích thước.

Trong cảnh xe tự hành, pruning có thể làm mất chi tiết của vật thể nhỏ ở xa nếu chỉ vài Gaussian có opacity thấp đóng góp vào vùng ảnh đó. Việc tái tạo cũng có thể khó ngay từ đầu nếu vật thể chỉ chiếm vài pixel trong ảnh huấn luyện hoặc có ít điểm SfM.

## Tài liệu tham khảo

- [Bài báo 3D Gaussian Splatting](../Sources/3d_gaussian_splatting_low.pdf)
- [Mã nguồn tham chiếu của nhóm tác giả](https://github.com/graphdeco-inria/gaussian-splatting)
