function chIndexSym = getChIndexSym(indexNRDL, nSubC, CPlen, iSym)

if nSubC == 512 && CPlen == 128
    nSymSlot = 48;
    iSlot = floor(iSym / 24);
    freqBinSize = 12*24;
elseif nSubC == 512
    nSymSlot = 56;
    iSlot = floor(iSym / 28);
    freqBinSize = 12*24;
elseif nSubC == 1024
    nSymSlot = 28;
    iSlot = floor(iSym / 14);
    freqBinSize = 12*51;
elseif nSubC == 2048
    nSymSlot = 14;
    iSlot = floor(iSym / 7);
    freqBinSize = 12*106;
end


chIndexSlot = union(union(union(union(union(indexNRDL.indexPDSCH{iSlot+1}, ...
    indexNRDL.indexPDSCHDMRS{iSlot+1}), ...
    indexNRDL.indexPDSCHPTRS{iSlot+1}), ...
    indexNRDL.indexPDCCH{iSlot+1}), ...
    indexNRDL.indexSSBurst{iSlot+1}), ...
    indexNRDL.indexCSIRS{iSlot+1});

chIndexSym = chIndexSlot(chIndexSlot > mod(iSym, nSymSlot/2)*freqBinSize & ...
    chIndexSlot <= mod(iSym+1, nSymSlot/2)*freqBinSize);

end