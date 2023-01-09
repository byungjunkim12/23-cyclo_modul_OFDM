clear;
clc;
% close all

% signal parameters
% fs = 100e6;
% snrVec = 1:1:2;

% SSCA parameters
N = 4096;
tau_max = 300;
est_len = 16;

N_hop = 1;
tau_hop = 1;


%% Load the signal
% data_dir = '/tmp/fsk4_1MHz_sps_4';
% save_dir = '/tmp/ssca_ofdm/';

% for file_index = 1 : numel(fileinfo)
in = read_complex_binary('/tmp/test_ofdm/ax_2_0_10_20_0.4_1.32cf');
in = in(1201:8000);

index_matrix = (1:est_len)' + (0:(numel(in)-est_len));
in_matrix = in(index_matrix);

tau_vector = 0:tau_hop:tau_max;
CAF = zeros(N, numel(tau_vector));

for tau_index = 1:numel(tau_vector)
    tau = tau_vector(tau_index);

    if est_len == 1
        in_corr = in_matrix(1:end-tau) .* conj(in_matrix(1+tau:end));
    else
        in_corr = (sum(in_matrix(:, 1:end-tau) .* conj(in_matrix(:, 1+tau:end)), 1));
    end

    nFeat = numel(1 : N_hop : (numel(in_corr) - N));
    in_spec = zeros(N, nFeat);
    for i = 1:nFeat
        in_spec(:, i) = fft(in_corr(N_hop*(i-1)+1 : N_hop*(i-1)+N));
    end
    in_spec_mean = mean(abs(in_spec), 2);
    CAF(:, tau_index) = in_spec_mean;
end

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

% subplot(2,2,3);
% plot(2:N, CAF(2:end, 65));
% 
% subplot(2,2,4);
% plot(2:N, CAF(2:end, 66));

% for tau = 1:tau_max
%
% end