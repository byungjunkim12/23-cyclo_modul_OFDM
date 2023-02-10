function featMat = extractModFeat(inputIQ, nSubC, CPlen, firstIndex, nSym)

featMat = zeros(nSubC, nSym);
startIndex = firstIndex + floor(CPlen/2);
symFreq = fft(inputIQ(startIndex : startIndex + nSubC-1));

for iSym = 1 : nSym
    nextSymFreq = fft(inputIQ(startIndex + iSym * (nSubC+CPlen) : ...
       startIndex + iSym * (nSubC+CPlen) + nSubC-1));
    
    featAbs = abs(symFreq);
    featPhase = angle(symFreq) - angle(nextSymFreq);
    featMat(:, iSym) = featAbs .* exp(1j*featPhase);
    symFreq = nextSymFreq;
end
end