data = [ 0, 987, 537, 181, 49, 36, 14, 20, 8, 96];
x = 0.1:0.1:1;
f = figure;
bar(x,data);
%b.FaceColor = 'flat';
%b.CData(2,:) = [.5 0 .5];
xticks([0.1:0.1:1])
xticklabels({'0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','> 1'})
xlabel('Duration in sec')
ylabel('Consecutive fixation count')