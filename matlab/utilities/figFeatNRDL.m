function figFeatNRDL(inputIQ, indexNRDL, nSubC, CPlen, nSym, angleMod, firstIndex, firstSymIndex)
nFreqRB = 12;
chIndexSym = cell(nSym+1, 1);
if nSubC == 512 && CPlen == 128 % BWP index [1, 512] = FFT index [368-512, 1-144]
    nSymSlot = 48;
    maxnRB = 24;
    indexTranMat = [369, 512, 1, 144];
elseif nSubC == 512 % CPlen = {36, 52}; BWP index [1, 512] = FFT index [368-512, 1-144]
    nSymSlot = 56;
    maxnRB = 24;
    indexTranMat = [369, 512, 1, 144];
elseif nSubC == 1024 % CPlen == {88, 72}; BWP index [1, 1024] = FFT index [719-1024, 1-306]
    nSymSlot = 28;
    maxnRB = 51;
    indexTranMat = [719, 1024, 1, 306];
elseif nSubC == 2048 % CPlen == {160, 144}; BWP index [1, 2048] = FFT index [1413-2048, 1-636]
    nSymSlot = 14;
    maxnRB = 106;
    indexTranMat = [1413, 2048, 1, 636];
end
freqBinSize = nFreqRB * maxnRB;
symIndexVec = firstSymIndex+(0:nSym)';
symLenVec = ones(size(symIndexVec)) * (nSubC+CPlen);
if CPlen ~= 128
    symLenVec = symLenVec + (mod(symIndexVec, (nSymSlot/2)) == 0) * 16;
end

if CPlen ~= 128
    longSymVec = find(mod(symIndexVec, (nSymSlot/2)) == 0);
    tempCumulIndex = [0; cumsum(symLenVec)];
    addLongCP = reshape(((0:15)' + tempCumulIndex(longSymVec)'), [16*(numel(longSymVec)), 1]);
    inputCrop = inputIQ(firstIndex+setdiff((0:(sum(symLenVec)-1)), addLongCP));
else
    inputCrop = inputIQ(firstIndex+(0:(sum(symLenVec)-1)));
end

lastSymIndex = firstSymIndex + nSym;
for iSym = 0 : (lastSymIndex - firstSymIndex)
    chIndexSym{iSym+1} = mod(getChIndexSym(indexNRDL, nSubC, CPlen, iSym+firstSymIndex)-1, ...
        freqBinSize)+1;
end

featAbs = NaN * zeros(freqBinSize, nSym);
featPh = NaN * zeros(freqBinSize, nSym);
% feat = zeros(freqBinSize, nSym);
activeSymIndex = -1 * ones(freqBinSize, 1);
activeSymIndex(chIndexSym{1}) = 0;
% ActivefeatAbs = zeros(nSubC, 1);
% ActivefeatPh = zeros(nSubC, 1);

prevSymFreq = fft(inputCrop(((CPlen/2):(nSubC+(CPlen/2)-1))));
% prevSymFreq = fft(inputIQ((firstIndex+(CPlen/2):(firstIndex+nSubC+(CPlen/2)-1))));

prevSymIndex = prevSymFreq([indexTranMat(1):indexTranMat(2), ...
    indexTranMat(3):indexTranMat(4)]);

prevSymAbs = NaN * zeros(freqBinSize, 1);
prevSymPh = NaN * zeros(freqBinSize, 1);

prevSymAbs(chIndexSym{1}) = abs(prevSymIndex(chIndexSym{1}));
prevSymPh(chIndexSym{1}) = angle(prevSymIndex(chIndexSym{1}));

for iSym = 1 : nSym
    symFreq = fft(inputCrop(((CPlen/2) + iSym*(nSubC+CPlen)) ...
        : (nSubC+(CPlen/2)-1 + iSym*(nSubC+CPlen))));
    %     symFreq = fft(inputIQ((firstIndex+(CPlen/2) + iSym*(nSubC+CPlen)) ...
    %         : (firstIndex+nSubC+(CPlen/2)-1 + iSym*(nSubC+CPlen))));
    symIndex = symFreq([indexTranMat(1):indexTranMat(2), ...
        indexTranMat(3):indexTranMat(4)]);

    symAbs = NaN * zeros(freqBinSize, 1);
    symPh = NaN * zeros(freqBinSize, 1);
    symAbs(chIndexSym{iSym+1}) = abs(symIndex(chIndexSym{iSym+1}));
    symPh(chIndexSym{iSym+1}) = angle(symIndex(chIndexSym{iSym+1}));

    validFeatIndices = intersect(find(~isnan(prevSymAbs)), find(~isnan(symAbs)));
    featAbs([validFeatIndices + activeSymIndex(validFeatIndices)*freqBinSize]) = ...
        prevSymAbs(validFeatIndices);
    featPh([validFeatIndices + activeSymIndex(validFeatIndices)*freqBinSize]) = ...
        symPh(validFeatIndices) - prevSymPh(validFeatIndices);

%     mod(symPh(validFeatIndices) - prevSymPh(validFeatIndices), pi/4);

    %     featAbs(:, iSym) = abs(symFreq);
    %     featPh(:, iSym) = angle(nextSymFreq) - angle(symFreq);

    validInputIndices = find(~isnan(symAbs));
    prevSymAbs(validInputIndices) = symAbs(validInputIndices);
    prevSymPh(validInputIndices) = symPh(validInputIndices);
    activeSymIndex(validInputIndices) = iSym;

    %     subplot(1, 2, 1)

    %     subplot(1, 2, 2)
    %     scatter(real(feat(pilotSubC)), imag(feat(pilotSubC)), 200, 'filled'); hold on;

%         featInputHist(((iSym-1)*nSubC+1) : (iSym*nSubC)) = symPh(validFeatIndices) - prevSymPh(validFeatIndices);
%         featInputHist(((iSym-1)*numel(pilotSubC)+1) : (iSym*numel(pilotSubC))) = featPh(pilotSubC);

end

f = figure;
if angleMod
    featPh = mod(featPh, pi/4);
    
    %     tempFeatPh = featPh;
    %     featPh = tempFeatPh;
    featPh(featPh < atan(1/17)) = NaN*featPh(featPh < atan(1/17));
    featPh(featPh > pi/2-atan(1/17)) = NaN*featPh(featPh > pi/2-atan(1/17));

    featPh(featPh > atan(15/17) & featPh < atan(17/15)) = ...
        NaN * featPh(featPh > atan(15/17) & featPh < atan(17/15));

    featPh(featPh > pi/4) = pi/2 - featPh(featPh > pi/4);
    featPh(featPh > pi/8) = pi/4 - featPh(featPh > pi/8);

end
feat = featAbs .* exp(1j*featPh);
scatter(real(feat), imag(feat), 50, 'filled');

% if removeNull
%     [~, sortSubC] = sort(mean(featAbs, 2));
%     subCwoNull = sortSubC(nNullSubC+1:end);
%     featRemoveNull = feat(subCwoNull, :);
%     scatter(real(featRemoveNull), imag(featRemoveNull), 50, 'filled');
% else
%     scatter(real(feat), imag(feat), 100, 'filled');
% end
xlabel('Real'); ylabel('Imag')
ax = gca;
ax.FontSize = 24;
f.Position = [600 600 1000 600];
% xlim([0 5]); ylim([0 3])

% featInputHist(featInputHist > pi) = featInputHist(featInputHist > pi) - 2*pi;
% featInputHist(featInputHist < -pi) = featInputHist(featInputHist < -pi) + 2*pi;

figure;
subplot(1, 2, 1)
histogram(featPh, 25);
% featInputHist(featInputHist < 0) = featInputHist(featInputHist < 0) + pi;
subplot(1, 2, 2)
histogram(featAbs, 25);

end