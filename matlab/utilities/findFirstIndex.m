function [firstIndex, firstSymIndex] = findFirstIndex(inputIQ, nSubC, CPlen)
% for NRDL, numel(inputIQ) = 15360 + {0, 2208, 4400, 8794}
% (4 additional symbols)
if nSubC == 512 || nSubC == 1024 || nSubC == 2048
    nHalfSlotInput = floor(numel(inputIQ) / 15360);
    if nSubC == 512
        nSymSlot = 56;
    elseif nSubC == 1024
        nSymSlot = 28;
    elseif nSubC == 2048
        nSymSlot = 14;
    end
    nSymInput = nSymSlot * nHalfSlotInput;
end

indexMat = (1:CPlen)' + (0:(numel(inputIQ)-CPlen-1));
inputMat = inputIQ(indexMat);

inputCorr = (sum(inputMat(:, 1:end-nSubC) .* conj(inputMat(:, 1+nSubC:end)), 1));
if (nSubC == 512 && CPlen == 36) || nSubC == 1024 || nSubC == 2048
    pklocsMat = NaN*zeros((nSymInput/(2*nHalfSlotInput))+2, nHalfSlotInput);
    for iSlot = 1:nHalfSlotInput
        [~, pklocs] = findpeaks(abs(inputCorr(1+(iSlot-1)*15360 : ...
            (iSlot*15360 + mod(numel(inputIQ), 15360)) - (nSubC+CPlen))), ...
            'MinPeakDistance', floor(0.9*(nSubC+CPlen)), ...
            'NPeaks', (nSymInput/(2*nHalfSlotInput))+3);

        pklocsSym = floor(pklocs' / (nSubC+CPlen));
        %         pklocsSym = pklocsSym(2:1+min(numel(pklocsSym), size(pklocsMat, 1)));
        pkValidIndices = find(pklocsSym <= size(pklocsMat, 1) & pklocsSym > 0);
        %         pklocsSym = pklocsSym(pkValidIndices);

        pklocsMat(pklocsSym(pkValidIndices), iSlot) = ...
            pklocs((1:min(numel(pkValidIndices), size(pklocsMat, 1)))+pkValidIndices(1)-1);
        nanIndex = find(isnan(pklocsMat(:, iSlot)));
        pklocsMat(setdiff(nanIndex, size(pklocsMat, 1))+1, iSlot) = nan;
        pklocsMat(setdiff(nanIndex, 1)-1, iSlot) = nan;
    end
    remainder = mean(mod(pklocsMat, nSubC+CPlen), 2, 'omitnan');
    symOffset = floor(mean(pklocsMat(1, :), 'omitnan') / (nSubC+CPlen));
else
    [~, pklocs] = findpeaks(abs(inputCorr), 'MinPeakDistance', floor(0.9*(nSubC+CPlen)));
    remainder = mod(pklocs, nSubC+CPlen);
end


if (nSubC == 512 && CPlen == 36) || nSubC == 1024 || nSubC == 2048
    remainderDiff = remainder(3:end) - remainder(1:end-2);

    [~, symIndexCandidate] = maxk(remainderDiff, 2);
    if abs(symIndexCandidate(1) - symIndexCandidate(2)) == 1 && ...
            (remainderDiff(symIndexCandidate(1)) > 10 && ...
            remainderDiff(symIndexCandidate(2)) > 10)
        %         firstSymIndex = min(symIndexCandidate(1), symIndexCandidate(2));
        [~, minStdIndex] = min(std(pklocsMat(symIndexCandidate, :), 0, 2));
        firstSymIndex = symIndexCandidate(minStdIndex);

    elseif abs(symIndexCandidate(1) - symIndexCandidate(2)) == nSymSlot/2-1 && ...
            (remainderDiff(symIndexCandidate(1)) > 10 && ...
            remainderDiff(symIndexCandidate(2)) > 10)
        firstSymIndex = max(symIndexCandidate(1), symIndexCandidate(2));
    else
        firstSymIndex = symIndexCandidate(1);
    end
    firstSymIndex = mod(-(firstSymIndex+symOffset), (nSymSlot/2));
    symVec = (1:numel(remainder))';

    shortSymVec = symVec(mod(symVec+firstSymIndex, (nSymSlot/2)) ~= 0);
    % keep in mind that remainder starts from (symIndex+symOffset)th symbol!
    % shortSymVec starts from ()
    remainder = remainder - floor((symVec+firstSymIndex+symOffset - 1) / (nSymSlot/2)) * 16;
    remainder = remainder(shortSymVec');
end

if numel(unique(remainder)) == numel(remainder)
    firstIndex = round(median(remainder, 'omitnan'));
else
    firstIndex = round(mode(remainder));
end

if nSubC == 512 || nSubC == 1024 || nSubC == 2048
    if firstSymIndex == 0
        firstIndex = mod((firstIndex-16-1), nSubC+CPlen)+1;
    else
        firstIndex = mod((firstIndex-1), nSubC+CPlen)+1;
    end
end
% figure; plot(abs(inputCorr(1:6000)))
% xlabel('Time sample, $n$', 'Interpreter', 'latex')
% ylabel('$R_{yy}(n,N)$', 'Interpreter', 'latex')
% ax = gca;
% ax.FontSize = 24;

end