% clear;
% clc;
% close all

% SCF parameters
filename = '/tmp/test_ofdm/NRDL_2_0_0_20_0.4_1.32cf';
N = 1024;
tau_max = 1200;
est_len = 8;

% N_hop = 1;
% tau_hop = 1;


%% Load the signal
inputIQ = read_complex_binary(filename);
inputIQ = inputIQ(15361:30720);

tic;
CAF = csFeat(inputIQ, N, tau_max, est_len);
toc;

figure;
subplot(1,2,1);
plot(1:tau_max, CAF(1, 2:end))
xlabel('Time difference (\tau)');
ylabel('$$|\hat{R}_{xx}(0;\tau)|$$', 'Interpreter', 'Latex');

[~, nSubC_est] = max(CAF(1, 20:end));
nSubC_est = nSubC_est+18;
% nSubC_est = 256;

subplot(1,2,2);
plot(1:N-1, CAF(2:end, nSubC_est+1));
xlim([0 4095]);
xlabel('Freq index (\alpha)');
ylabel('$$|\hat{R}_{xx}(x;N)|$$', 'Interpreter', 'Latex');

% Continuous d