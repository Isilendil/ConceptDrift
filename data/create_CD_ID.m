function [ID_ALL] = create_CD_ID(Num_section)
% n - dataset size
%

ID_ALL = [];
for i=1:20,
    
    ID = [];
    for j=1:5,
        ID = [ID randperm(Num_section)+length(ID)];
    end
    
    ID_ALL=[ID_ALL; ID];
end