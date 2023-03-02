function firstIndex = findFirstIndex(inputIQ, nSubC, CPlen)
indexMat = (1:CPlen)' + (0:(numel(inputIQ)-CPlen));
inputMat = inputIQ(indexMat);

inputCorr = (sum(inputMat(:, 1:end-nSubC) .* conj(inputMat(:, 1+nSubC:end)), 1));
[~, pklocs] = findpeaks(abs(inputCorr), 'MinPeakDistance', floor(0.9*(nSubC+CPlen)));
firstIndex = mode(mod(pklocs, nSubC+CPlen));
end