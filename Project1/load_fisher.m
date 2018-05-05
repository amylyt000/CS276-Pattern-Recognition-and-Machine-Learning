function [m_train, m_test, f_train, f_test, u_train, train, test] = load_fisher()
m_all = zeros(256^2, 88);
f_all = zeros(256^2, 85);
u_all = zeros(256^2, 4)
% male faces
for i = 1:88
    if i < 58
        impath = ['./face_data/male_face/face', sprintf('%03d.bmp',i-1)];
    end
    if i >= 58
        impath = ['./face_data/male_face/face', sprintf('%03d.bmp',i)];
    end
    face = imread(impath);
    m_all(:,i) = reshape(face, [256^2, 1]);
end
m_train = m_all(:, 1:78); % 65536*78
m_test = m_all(:, 79:88); % 65536*10
% m_mean = sum(m_train, 2)/78;

% female faces
for i = 1:85
    impath = ['./face_data/female_face/face', sprintf('%03d.bmp',i-1)];
    face = imread(impath);
    f_all(:,i) = reshape(face, [256^2, 1]);
end
f_train = f_all(:, 1:75); % 65536*75
f_test = f_all(:, 76:85); % 65536*10
% f_mean = sum(f_train, 2)/75;

% unknown faces
for i = 1:4
    impath = ['./face_data/unknown_face/face', sprintf('%03d.bmp',i-1)];
    face = imread(impath);
    u_all(:,i) = reshape(face, [256^2, 1]);
end
u_train = u_all; % 65536*4

% train sets and test sets
train = [m_train, f_train]; % 65536*153
test = [m_test, f_test]; % 65536*20
end

