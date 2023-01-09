clear;
clc;
% close all

% SCF parameters
filename = '/tmp/test_ofdm/ax_2_0_10_20_0.4_1.32cf';
N = 4096;
tau_max = 300;
est_len = 16;

% N_hop = 1;
% tau_hop = 1;


%% Load the signal
inputIQ = read_complex_binary(filename);
inputIQ = inputIQ(1201:8000);

CAF = cs_feature(inputIQ, N, tau_max, est_len);


figure;
subplot(1,2,1);
plot(1:tau_max, CAF(1, 2:end))
xlabel('Time difference (\tau)');
ylabel('$$|\hat{R}_{xx}(0;\tau)|$$', 'Interpreter', 'Latex');

[~, nSubC_est] = max(CAF(1, 20:end));
nSubC_est = nSubC_est+18;
nSubC_est = 256;

subplot(1,2,2);
plot(1:N-1, CAF(2:end, nSubC_est+1));
xlim([0 4095]);
xlabel('Freq index (\alpha)');
ylabel('$$|\hat{R}_{xx}(x;N)|$$', 'Interpreter', 'Latex');

