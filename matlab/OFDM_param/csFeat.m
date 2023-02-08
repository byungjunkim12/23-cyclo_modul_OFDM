function CAF = csFeat(inputIQ, FFTsize, tau, lenCPest)

indexMat = (1:lenCPest)' + (0:(numel(inputIQ)-lenCPest));
inputMat = inputIQ(indexMat);

if lenCPest == 1
    inputCorr = inputMat(1:end-tau) .* conj(inputMat(1+tau:end));
else
    inputCorr = (sum(inputMat(:, 1:end-tau) .* conj(inputMat(:, 1+tau:end)), 1));
end

% nFeat = numel(1 : Nhop : (numel(inputCorr) - FFTsize + 1));
inputSpec = zeros(FFTsize, (numel(inputCorr) - FFTsize + 1));
for i = 1:size(inputSpec, 2)
    inputSpec(:, i) = fft(inputCorr(i : (i-1)+FFTsize));
end
CAF = mean(abs(inputSpec), 2);

end