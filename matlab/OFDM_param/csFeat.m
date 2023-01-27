function CAF = csFeat(inputIQ, N, tauMax, lenCP_est, N_hop, tauHop)
if ~exist('N_hop', 'var')
    N_hop = 1;
end
if ~exist('tau_hop', 'var')
    tauHop = 1;
end

indexMat = (1:lenCP_est)' + (0:(numel(inputIQ)-lenCP_est));
inputMat = inputIQ(indexMat);
tauVec = 0:tauHop:tauMax;
CAF = zeros(N, numel(tauVec));

for tauIndex = 1:numel(tauVec)
    tau = tauVec(tauIndex);

    if lenCP_est == 1
        inputCorr = inputMat(1:end-tau) .* conj(inputMat(1+tau:end));
    else
        inputCorr = (sum(inputMat(:, 1:end-tau) .* conj(inputMat(:, 1+tau:end)), 1));
    end

    nFeat = numel(1 : N_hop : (numel(inputCorr) - N + 1));
    inputSpec = zeros(N, nFeat);
    for i = 1:nFeat
        inputSpec(:, i) = fft(inputCorr(N_hop*(i-1)+1 : N_hop*(i-1)+N));
    end
    inputSpecMean = mean(abs(inputSpec), 2);
    CAF(:, tauIndex) = inputSpecMean;
end

end

