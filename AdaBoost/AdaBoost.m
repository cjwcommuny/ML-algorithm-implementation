function Alpha = AdaBoost(M,x,y)
%
    N_data = length(x);
    D = zeros(M+1, N_data);
    D(1,:) = 1/N_data;%weight initialization
    pivot = zeros(M,1);
    G = @classifier;
    ErrorRate = ones(M,1);
    Alpha = zeros(M,1);
    Z = zeros(M,1);
    for m = 1:M
        [ErrorRate(m), pivot(m)] = PrimaryClassifier(x, y, D(m,:));
        Alpha(m) = 1/2 * log((1-ErrorRate(m))/ErrorRate(m));
        Z(m) = sum(sum(D(m,:)'.*exp(-Alpha(m)*y.*G(x,pivot(m)))));
        D(m+1,:) = 1/Z(m)*D(m,:)' .* exp(-Alpha(m)*y.*G(x,pivot(m)));
    end

end

% x = [0 1 2 3 4 5 6 7 8 9]';
% y = [1 1 1 -1 -1 -1 1 1 1 -1]';
function y = classifier(x, v)
    y = (x<v) - (x>v);
end