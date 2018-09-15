function [e, pivot] = PrimaryClassifier(x, y, weight)
%
    maximum = max(x);
    minmum = min(x);
    step = 0.5;
    MinPivot = minmum;
    MinError = 1;
    for v = minmum:step:maximum
        ErrorTmp = sum(sum(weight'.*(y ~= classifier(x,v))));
        if (ErrorTmp < MinError)
            MinPivot = v;
            MinError = ErrorTmp;
        end
    end
    pivot = MinPivot;
    e = MinError;
end

function y = classifier(x, v)%not return bool vector
    y = (x<v) - (x>v);
end