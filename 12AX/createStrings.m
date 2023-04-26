function vector = createStrings(strings,duration)
% 1 A X 2 B Y C D
if nargin < 2
    duration = 300;
end
vector = [];
for i=1:length(strings)
    let = strings(i);
    switch let
        case '1'
            h=[1,0,0,0,0,0,0,0];
        case 'A'
            h=[0,1,0,0,0,0,0,0];
%             h=[1,1,0,0,0,0,0,0];
        case 'X'
            h=[0,0,1,0,0,0,0,0];
%             h=[1,0,1,0,0,0,0,0];
        case '2'
            h=[0,0,0,1,0,0,0,0];
        case 'B'
            h=[0,0,0,0,1,0,0,0];
        case 'Y'
            h=[0,0,0,0,0,1,0,0];
        case 'C'
            h=[0,0,0,0,0,0,1,0];
        case 'D'
            h=[0,0,0,0,0,0,0,1];
    end
    vector = [vector; repmat(h,duration,1)];
            
end