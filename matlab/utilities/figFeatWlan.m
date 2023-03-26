function figFeatWlan(inputIQ, nSubC, CPlen, firstIndex, nSym, removeNull, angleMod)
% pilotSubC = [23, 49, 91, 117, 141, 167, 209, 235];
% pilotSubC = [22, 23, 24, 48, 49, 50, 90, 91, 92, 116, 117, 118,...
%     140, 141, 142, 166, 167, 168, 208, 209, 210, 234, 235, 236];
% featInputHist = zeros(numel(pilotSubC)*nSym, 1);
if nSubC == 64
    nNullSubC = 8;
elseif nSubC == 256
    nNullSubC = 14;
end

featAbs = zeros(nSubC, nSym);
featPh = zeros(nSubC, nSym);
feat = zeros(nSubC, nSym);
for iSym = 1 : nSym
    symbolsFreq = fft(inputIQ((firstIndex+(CPlen/2) + (iSym-1)*(nSubC+CPlen)) ...
        : (firstIndex+nSubC+(CPlen/2)-1 + (iSym-1)*(nSubC+CPlen))));
    nextSymbolsFreq = fft(inputIQ((firstIndex+(CPlen/2) + iSym*(nSubC+CPlen)) ...
        : (firstIndex+nSubC+(CPlen/2)-1 + iSym*(nSubC+CPlen))));
    featAbs(:, iSym) = abs(symbolsFreq);
    featPh(:, iSym) = angle(nextSymbolsFreq) - angle(symbolsFreq);
    if angleMod
        featPh(:, iSym) = mod(featPh(:, iSym), pi/2);
        tempFeatPh = featPh(:, iSym);
        tempFeatPh(tempFeatPh > pi/4) = pi/2 - tempFeatPh(tempFeatPh > pi/4);
        featPh(:, iSym) = tempFeatPh;
    end
    feat(:, iSym) = featAbs(:, iSym) .* exp(1j*featPh(:, iSym));
    %     subplot(1, 2, 1)
    
    %     subplot(1, 2, 2)
    %     scatter(real(feat(pilotSubC)), imag(feat(pilotSubC)), 200, 'filled'); hold on;

    %     featInputHist(((iSym-1)*nSubC+1) : (iSym*nSubC)) = featPh;
    %     featInputHist(((iSym-1)*numel(pilotSubC)+1) : (iSym*numel(pilotSubC))) = featPh(pilotSubC);
end

f = figure;
if removeNull
    [~, sortSubC] = sort(mean(featAbs, 2));
    subCwoNull = sortSubC(nNullSubC+1:end);
    featRemoveNull = feat(subCwoNull, :);
    scatter(real(featRemoveNull), imag(featRemoveNull), 50, 'filled');
else
    scatter(real(feat), imag(feat), 100, 'filled');
end
xlabel('Real'); ylabel('Imag')
ax = gca;
ax.FontSize = 24; 
f.Position = [600 600 1000 600];
% xlim([0 5]); ylim([0 3])

% featInputHist(featInputHist > pi) = featInputHist(featInputHist > pi) - 2*pi;
% featInputHist(featInputHist < -pi) = featInputHist(featInputHist < -pi) + 2*pi;

% figure;
% subplot(1, 2, 1)
% histogram(featInputHist, 25);
% featInputHist(featInputHist < 0) = featInputHist(featInputHist < 0) + pi;
% subplot(1, 2, 2)
% histogram(featInputHist, 12);

end