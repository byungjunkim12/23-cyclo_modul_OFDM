function CFOest = estCFO(inputIQ, nSubC, lenCP, firstIndexSymbol, samplingRate)
lenOFDMSymbol = nSubC + lenCP;
nOFDMSymbol = floor(numel(inputIQ(firstIndexSymbol + (lenCP/4):end)) / (lenOFDMSymbol));
phDiffSymbol = zeros(1, nOFDMSymbol);

for i = 1 : nOFDMSymbol
    phDiffSymbol(i) = mean(angle(inputIQ(firstIndexSymbol + (lenCP/4) + (i-1)*lenOFDMSymbol + nSubC :...
        firstIndexSymbol + (lenCP/4) + (i-1)*lenOFDMSymbol + nSubC + (lenCP/2) - 1) ./ ...
        inputIQ(firstIndexSymbol + (lenCP/4) + (i-1)*lenOFDMSymbol :...
        firstIndexSymbol + (lenCP/4) + (i-1)*lenOFDMSymbol + (lenCP/2) - 1)));
end
CFOest = mean(unwrap(phDiffSymbol)) / (2*pi*nSubC) * samplingRate;
end