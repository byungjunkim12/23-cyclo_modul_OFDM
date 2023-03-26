function indexNRDLCell = indexNRDLshaping(indexNRDLMetadata)
fieldNames= fieldnames(indexNRDLMetadata);
for fieldIndex = 1 : numel(fieldNames)
    fieldName = fieldNames{fieldIndex};
    if iscell(indexNRDLMetadata.(fieldName))
        indexNRDLCell.(fieldName) = indexNRDLMetadata.(fieldName);
        for slotIndex = 1 : numel(indexNRDLCell.(fieldName))
            indexNRDLCell.(fieldName){slotIndex} = ...
                cell2mat(indexNRDLCell.(fieldName){slotIndex});
        end
    else
        indexNRDLCell.(fieldName) = num2cell(indexNRDLMetadata.(fieldName), 2);
    end
end
 
end