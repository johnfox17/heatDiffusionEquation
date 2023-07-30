clear all;
close all;
addpath('../data/')
SOL_PDDO = table2array(readtable("SOL_PDDO.csv"));
coords = table2array(readtable("coords.csv"));
times = table2array(readtable("time.csv"));

SOL_ANALYTICAL_0_7 = exp(-0.7).*sin(pi*coords(:,1)).*cos(pi*coords(:,2)); 
SOL_ANALYTICAL_0_7 = reshape(SOL_ANALYTICAL_0_7, [21 21]);
SOL_ANALYTICAL_0_5 = exp(-0.5).*sin(pi*coords(:,1)).*cos(pi*coords(:,2)); 
SOL_ANALYTICAL_0_5 = reshape(SOL_ANALYTICAL_0_5, [21 21]);
SOL_ANALYTICAL_0_3 = exp(-0.3).*sin(pi*coords(:,1)).*cos(pi*coords(:,2));
SOL_ANALYTICAL_0_3 = reshape(SOL_ANALYTICAL_0_3, [21 21]);
SOL_ANALYTICAL_0_1 = exp(-0.1).*sin(pi*coords(:,1)).*cos(pi*coords(:,2));
 SOL_ANALYTICAL_0_1 = reshape(SOL_ANALYTICAL_0_1, [21 21]);

SOL_PDDO_0_7= reshape(SOL_PDDO(14,:), [21 21]);
SOL_PDDO_0_5= reshape(SOL_PDDO(10,:), [21 21]);
SOL_PDDO_0_3= reshape(SOL_PDDO(6,:), [21 21]);
SOL_PDDO_0_1= reshape(SOL_PDDO(2,:), [21 21]);

figure;
hold on;
i=13;

plot(SOL_PDDO_0_7(:,i),'b-o')
plot(SOL_ANALYTICAL_0_7(:,i), 'b-*')

plot(SOL_PDDO_0_5(:,i),'r-o')
plot(SOL_ANALYTICAL_0_5(:,i), 'r-*')

plot(SOL_PDDO_0_3(:,i),'k-o')
plot(SOL_ANALYTICAL_0_3(:,i), 'k-*')

plot(SOL_PDDO_0_1(:,i),'g-o')
plot(SOL_ANALYTICAL_0_1(:,i), 'g-*')
legend('PDDO - 0.7', 'Analytical - 0.7', 'PDDO - 0.5', 'Analytical - 0.5', ...
    'PDDO - 0.3', 'Analytical - 0.3', 'PDDO - 0.1', 'Analytical - 0.1')
title('x_2 = 0.6')

i=9;
figure;
hold on;
plot(SOL_PDDO_0_7(i,:),'b-o')
plot(SOL_ANALYTICAL_0_7(i,:), 'b-*')
plot(SOL_PDDO_0_5(i,:),'r-o')
plot(SOL_ANALYTICAL_0_5(i,:), 'r-*')
plot(SOL_PDDO_0_3(i,:),'k-o')
plot(SOL_ANALYTICAL_0_3(i,:), 'k-*')
plot(SOL_PDDO_0_1(i,:),'g-o')
plot(SOL_ANALYTICAL_0_1(i,:), 'g-*')
legend('PDDO - 0.7', 'Analytical - 0.7', 'PDDO - 0.5', 'Analytical - 0.5', ...
    'PDDO - 0.3', 'Analytical - 0.3', 'PDDO - 0.1', 'Analytical - 0.1')
title('x_1 = 0.4')
