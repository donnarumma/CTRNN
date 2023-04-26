function [totalOut, f, isFix, outOnlyAsym]=plotPhasePortrait(net,externalInput)
%function [out, f, isFix, outOnlyAsym]=plotPhasePortrait(net,externalInput)
     
     f=figure;
     %set(f, 'Visible', 'off');
     count_i=1;
     
     for i=0:0.1:1
        count_j=1;
        for j=0:0.1:1
            net.initialOutValues=[i, j];
            [out, isFix, outOnlyAsym]=runCTRNN_AsymWay(net,externalInput);
            totalOut{count_i,count_j}=out;
            plot(out(:,1),out(:,2),'r');
            hold on;
            count_j=count_j+1;
        end
        count_i=count_i+1;
    end
end