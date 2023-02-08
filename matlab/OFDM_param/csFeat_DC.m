function CAF_DC = csFeat_DC(inputIQ, tauVec, lenCPest)

indexMat = (1:lenCPest)' + (0:(numel(inputIQ)-lenCPest));
inputMat = inputIQ(indexMat);
CAF_DC = zeros(1, numel(tauVec));

for tauIndex = 1:numel(tauVec)
    tau = tauVec(tauIndex);

    if lenCPest == 1
        inputCorr = inputMat(1:end-tau) .* conj(inputMat(1+tau:end));
    else
        inputCorr = (sum(inputMat(:, 1:end-tau) .* conj(inputMat(:, 1+tau:end)), 1));
    end
    CAF_DC(tauIndex) = mean(abs(inputCorr));
end

end