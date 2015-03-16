D = csvread('eyes.csv');

t = D(1,1);
T_on = 0;
T_off = 0;

for k = 2: size(D,1)
    
    if D(k-1,2) ~= D(k,2)
        delta_t = D(k,2) - t;
        t = D(k,2);
        
    end
    
    
    
end