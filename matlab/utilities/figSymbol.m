function figSymbol(inputIQ, nSubC, CPlen, firstIndex, nSym)
figure;
for iSym = 1 : nSym
    symbolsFreq = fft(inputIQ((firstIndex+CPlen+(iSym-1)*(nSubC+CPlen)) ...
        : (firstIndex+nSubC+CPlen-1+(iSym-1)*(nSubC+CPlen))));
    scatter(real(symbolsFreq), imag(symbolsFreq), 50, 'filled')
    hold on
end
xlabel('Real'); ylabel('Imag')

figure;
plot(abs(symbolsFreq));
end