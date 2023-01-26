function firstIndex = getFirstIndexOFDMSymbol(inputIQ, nSubC, lenCP)
indexMat = (1:lenCP)' + (0:(numel(inputIQ)-lenCP));
inputMat = inputIQ(indexMat);
inputCorr = (sum(inputMat(:, 1:end-nSubC) .* conj(inputMat(:, 1+nSubC:end)), 1));

[~, peakIndices] = findpeaks(abs(inputCorr), "MinPeakDistance", nSubC);
firstIndex = mode(rem(peakIndices-1, (nSubC+lenCP))+1);
end