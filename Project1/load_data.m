function[train, test] = load_data()
%train = zeros(256^2, 150);
%test = zeros(256^2, 27);
face_matrix = zeros(256^2, 177);
for i = 1:177
    if i < 104
        impath = ['./face_data/face/face', sprintf('%03d.bmp',i-1)];
    end
    if i >= 104
        impath = ['./face_data/face/face', sprintf('%03d.bmp',i)];
    end
    face = imread(impath);
    face_matrix(:,i) = reshape(face, [256^2, 1]);
end
train = face_matrix(:, 1:150);
test = face_matrix(:, 151:177);



    
