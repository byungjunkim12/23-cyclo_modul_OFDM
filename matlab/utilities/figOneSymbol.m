function figOneSymbol(inputIQ, nSubC, lenCP, lenPreamble)
    symbols_freq = fft(inputIQ((lenPreamble+lenCP+1) : (lenPreamble+nSubC+lenCP)));
    figure;
    scatter(real(symbols_freq), imag(symbols_freq), 200, 'filled')
end