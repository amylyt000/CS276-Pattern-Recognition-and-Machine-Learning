function paint_histogram(row,N_pos,N_neg)
% row = round(row);
% for i = 1
%     figure;
%     [n1,x1] = hist(row(i, 1:N_pos),[min(row(i, :)) : 1 : max(row(i, :))]);
%     h1=bar(x1,n1,'hist');
%     set(h1,'facecolor','c');
%     hold on;
%     [n2,x2] = hist(row(i, N_pos+1:N_pos+N_neg),[min(row(i, :)) : 1 : max(row(i, :))]);
%     h2=bar(x2,n2,'hist');
%     set(h2,'facecolor','m')
%     hold off;
%     legend('Pos(face)', 'Neg(nonface)');
%     alpha(0.5);
% end
for i = 1
    figure;
    histogram(row(i,1:N_pos));
    hold on;
    histogram(row(i,N_pos+1:end));
    hold off;
    legend('Pos(face)','Neg(non-face)');
    alpha(0.5);

end