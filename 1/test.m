
figure(1)

plot(elastic.VarName1,elastic.VarName4, linewidth = 2);
xlim([0,10])
title("SSE Elastic")

figure(2)

plot(lasso.VarName1,lasso.VarName4, linewidth = 2);
xlim([0,10])
title("SSE Lasso")


figure(3)
plot(ridge.VarName1,ridge.VarName4, linewidth = 2);
xlim([0,10])
title("SSE Ridge")
