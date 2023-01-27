function CFOest = estCFO(inputIQ, nSubC, lenCP, firstIndexSymbol, samplingRate)
lenOFDMSymbol = nSubC + lenCP;
nOFDMSymbol = floor(numel(inputIQ(firstIndexSymbol + (lenCP/4):end)) / (lenOFDMSymbol));
phDiffSymbol = zeros(1, nOFDMSymbol);

for i = 1 : nOFDMSymbol
    phDiffSymbol(i) = mean(angle(inputIQ(firstIndexSymbol + (lenCP/4) + (i-1)*lenOFDMSymbol + nSubC + 1 :...
        firstIndexSymbol + (lenCP/4) + (i-1)*lenOFDMSymbol + nSubC + (lenCP/2)) ./ ...
        inputIQ(firstIndexSymbol + (lenCP/4) + (i-1)*lenOFDMSymbol + 1 :...
        firstIndexSymbol + (lenCP/4) + (i-1)*lenOFDMSymbol + (lenCP/2))));
end
CFOest = mean(unwrap(phDiffSymbol)) / (2*pi*nSubC) * samplingRate;
end