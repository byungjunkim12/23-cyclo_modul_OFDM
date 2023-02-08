SNRvec = 0:2:20;

acc_nSubC_wlan_AWGN = [.9655, .9944, 1, 1, 1, 1, 1, 1, 1, 1, 1];
acc_nSubC_NRDL_AWGN = [.9933, .9983, 1, 1, 1, 1, 1, 1, 1, 1, 1];

acc_CPlen_wlan_AWGN = [.944, .9769, .9949, 1, 1, .9987, 1, .9974, 1, .9987, .9987];
acc_CPlen_NRDL_AWGN = [.9428, .9773, .9949, .9993, .9993, .9982, 1, .9978, 1, .9989, .9989];


figure;
plot(SNRvec, acc_nSubC_wlan_AWGN, '-xb', 'LineWidth', 2, 'MarkerSize', 20);
hold on;
plot(SNRvec, acc_nSubC_wlan_AWGN, '-ob', 'LineWidth', 2, 'MarkerSize', 20);
hold on;

plot(SNRvec, acc_CPlen_wlan_AWGN, '-xr', 'LineWidth', 2, 'MarkerSize', 20);
hold on;
plot(SNRvec, acc_nSubC_wlan_AWGN, '-or', 'LineWidth', 2, 'MarkerSize', 20);

ylim([0 1]);
xlabel('SNR (dB)')
ylabel('Accuracy')
grid on;
legend(["Est. of # of subcarriers of Wi-Fi 6 signals", ...
    "Joint Est. of # of subcarriers and CP length of Wi-Fi 6 signals", ...
    "Est. of # of subcarriers of 5G signals", ...
    "Joint Est. of # of subcarriers and CP length of 5G signals"])
