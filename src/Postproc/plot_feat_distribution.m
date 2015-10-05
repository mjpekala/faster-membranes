function plot_feat_distribution(X, Y)

[fp,xp] = ksdensity(X(Y==1));
[fn,xn] = ksdensity(X(Y==0));

figure('Position', [100 100 1200 400]); 
ha = tight_subplot(1, 3, [.03, .03]);

axes(ha(1));
[f,x] = ecdf(X(:));
plot(x, f);
grid on;

axes(ha(2));
plot(xp, fp, xn, fn);
grid on;
legend('membrane', 'non-membrane');

axes(ha(3));
boxplot(X(:), Y(:), 'labels', {'non-membrane', 'membrane'});
