
figure(1)



plot(ridgecrossV.VarName1,ridgecrossV.VarName3, linewidth = 2);
hold on
title("R^2")

figure(2)
plot(ridgecrossV.VarName1,ridgecrossV.VarName4, linewidth = 2);

title("SSE")

figure(3)



plot(ridgecrossV2old.VarName1,ridgecrossV2old.VarName3, linewidth = 2);
hold on
title("R^2")
xlim([0,10])

figure(4)
plot(ridgecrossV2old.VarName1,ridgecrossV2old.VarName4, linewidth = 2);
xlim([0,10])
title("SSE")