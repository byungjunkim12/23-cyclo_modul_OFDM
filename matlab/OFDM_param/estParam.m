% clear;
% clc;
% close all

% SCF parameters
% filename = '/tmp/NRDL/SCS_60/NRDL_Normal_64qam/NRDL_Normal_64_0_20_20_2_2.32cf';
folder = '/tmp/OFDM_param/NRDL/SCS_60/';
SNRvector = 0:2:20;
FFTSize = 4096;
tau_vec = [64, 256, 333, 667, 1333];
est_len = 8;
Nmax = 25;
% freqBinCell = [52, 58]; % wlanHT
% freqBinCell = [14, 15, 16]; % wlanHE
freqBinCell = [11, 13];

%% Load the signal
corrCount_nSubC = zeros(1, numel(SNRvector));
% corrCount_CPlen = zeros(numel(freqBinCell), numel(SNRvector));

folderList = dir(folder);
for folderIndex = 3 : 6
    folder_modul = folder + folderList(folderIndex).name + '/';
    files = dir(folder_modul + '*.32cf');
    for fileIndex = 1 : numel(files)
        filename = files(fileIndex).name;
        filenameDigits = regexp(filename, '\d*', 'match');
        fileSNR = str2double(filenameDigits{3});
        SNRIndex = (fileSNR/2)+1;

        inputIQ = read_complex_binary(folder_modul + filename);
%         inputIQ = inputIQ(1201:end);
        start_index = randi(20001);
        inputIQ = inputIQ(start_index:(start_index+20000-1));

        % tic;
        CAF_DC = csFeat_DC(inputIQ, tau_vec, est_len);
        [~, argmaxTau] = max(CAF_DC);
        if argmaxTau == 3
            corrCount_nSubC(SNRIndex) = corrCount_nSubC(SNRIndex) + 1;
        end
        maxTau = tau_vec(argmaxTau);
        CAF = csFeat(inputIQ, FFTSize, maxTau, est_len);
        [~, argmaxFreq] = max(CAF(freqBinCell));
        if contains(filename, 'Normal') && argmaxFreq == 2
            corrCount_CPlen(1, SNRIndex) = corrCount_CPlen(1, SNRIndex) + 1;
        elseif contains(filename, 'Extended') && argmaxFreq == 1
            corrCount_CPlen(2, SNRIndex) = corrCount_CPlen(2, SNRIndex) + 1;
        end

%         if contains(filename, 'Normal') && argmaxFreq == 3
%             corrCount_CPlen(1, SNRIndex) = corrCount_CPlen(1, SNRIndex) + 1;
%         elseif contains(filename, 'Medium') && argmaxFreq == 2
%             corrCount_CPlen(2, SNRIndex) = corrCount_CPlen(2, SNRIndex) + 1;
%         elseif contains(filename, 'Extended') && argmaxFreq == 1
%             corrCount_CPlen(3, SNRIndex) = corrCount_CPlen(3, SNRIndex) + 1;
%         end

        % toc;
    end
end
acc_nSubC = corrCount_nSubC / ((numel(folderList)-2) * numel(files) / numel(SNRvector));
acc_CPlen = corrCount_CPlen / ((numel(folderList)-2) * numel(files) / (numel(SNRvector) * numel(freqBinCell)));

% figure;
% % subplot(1,2,1);
% scatter(tau_vec, CAF_DC, 200, 'filled');
% xlim([1 tau_vec(end)])
% xlabel('Time difference (\tau)');
% ylabel('$$|\hat{R}_{xx}(0;\tau)|$$', 'Interpreter', 'Latex');

% [~, nSubC_est] = max(CAF_DC(1, 20:end));
% nSubC_est = nSubC_est+18;
% % nSubC_est = 256;
% 
% subplot(1,2,2);
% plot(1:N-1, CAF_DC(2:end, nSubC_est+1));
% xlim([1 N-1]);
% xlabel('Freq index (\alpha)');
% ylabel('$$|\hat{R}_{xx}(x;N)|$$', 'Interpreter', 'Latex');