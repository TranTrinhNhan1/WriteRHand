'''
Dùng tập train để train K-fold tìm bộ siêu tham số tốt nhất
Cần preprocess lại từ đầu vì tập train đã chuẩn hóa theo mean/std, nếu dùng thì sẽ gây data leakage trên các fold.
'''