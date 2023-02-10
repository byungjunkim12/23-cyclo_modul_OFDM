function figOneSymbol(inputIQ, nSubC, CPlen, firstIndex)
    symbols_freq = fft(inputIQ((firstIndex+CPlen+1) : (firstIndex+nSubC+CPlen)));
    figure;
    scatter(real(symbols_freq), imag(symbols_freq), 200, 'filled')
end