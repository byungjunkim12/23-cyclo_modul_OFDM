SNRvec = 0:2:20;

acc_nSubC_wlan_AWGN = [.9836, .9967, .9994, 1, 1, 1, 1, 1, 1, 1, 1];
acc_nSubC_NRDL_AWGN = [.9967, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

acc_CPlen_wlan_AWGN = [.9653, .9881, .9989, .9992, .9994, .9994, .9992, 1, 1, 1, 1];
acc_CPlen_NRDL_AWGN = [.9724, .995, .99, .995, 1, 1, 1, 1, 1, 1, 1];


figure;
plot(SNRvec, acc_nSubC_wlan_AWGN, '-xb', 'LineWidth', 2, 'MarkerSize', 20);
hold on;
plot(SNRvec, acc_CPlen_wlan_AWGN, '-ob', 'LineWidth', 2, 'MarkerSize', 20);
hold on;

plot(SNRvec, acc_nSubC_NRDL_AWGN, '-xr', 'LineWidth', 2, 'MarkerSize', 20);
hold on;
plot(SNRvec, acc_CPlen_NRDL_AWGN, '-or', 'LineWidth', 2, 'MarkerSize', 20);

ylim([0.8, 1]);
xlabel('SNR (dB)')
ylabel('Accuracy')
grid on;
legend(["Est. of # of subcarriers of Wi-Fi signals", ...
    "Joint Est. of # of subcarriers and CP length of Wi-Fi signals", ...
    "Est. of # of subcarriers of 5G signals", ...
    "Joint Est. of # of subcarriers and CP length of 5G signals"])



acc_nSubC_wlan_RICIAN = [.9237, .9606, .9815, .9916, .9947, .9986, .9989, .9992, 1, .9997, .9994];
acc_nSubC_NRDL_RICIAN = [.9083, .9617, .98, .995, .9883, .9983, 1, 1, 1, 1, 1];

acc_CPlen_wlan_RICIAN = [.8765, .9381, .9669, .9854, .9894, .9958, .9986, .9969, .9983, .9990, .9981];
acc_CPlen_NRDL_RICIAN = [.825, .905, .935, .965, .97, .99, 1, .99, .995, .995, 1];


figure;
plot(SNRvec, acc_nSubC_wlan_RICIAN, '-xb', 'LineWidth', 2, 'MarkerSize', 20);
hold on;
plot(SNRvec, acc_CPlen_wlan_RICIAN, '-ob', 'LineWidth', 2, 'MarkerSize', 20);
hold on;

plot(SNRvec, acc_nSubC_NRDL_RICIAN, '-xr', 'LineWidth', 2, 'MarkerSize', 20);
hold on;
plot(SNRvec, acc_CPlen_NRDL_RICIAN, '-or', 'LineWidth', 2, 'MarkerSize', 20);

ylim([0.8 1]);
xlabel('SNR (dB)')
ylabel('Accuracy')
grid on;
legend(["Est. of # of subcarriers of Wi-Fi signals", ...
    "Joint Est. of # of subcarriers and CP length of Wi-Fi signals", ...
    "Est. of # of subcarriers of 5G signals", ...
    "Joint Est. of # of subcarriers and CP length of 5G signals"])
