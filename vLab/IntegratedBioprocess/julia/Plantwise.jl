"""
    Plantwise()

- Julia version: 
- Author: Hua
- Date: 2022-09-29

# Arguments

# Examples

```jldoctest
julia>
```

"""

function Plantwide(x, p, t)
	# Written by Moo Sun Hong on Apr 4, 2017
	# For plant-wide model
	theta_M,theta_P,theta_C,u_F,u_Cin,u,tran = p;
	idx = min(length(tran[tran .<= t]), size(u_F)[2]);
	u_F = u_F[:, idx];
	u_Cin = u_Cin[:, idx];
	if t <= tran[2]
		feed = u[1,1:3]
		Cin = u[1,4:end]
	else
		feed = u[2, :]
		Cin = u[2,4:end]
	end
	p = [theta_M, feed, Cin]
	dxBdt = bioreactor_cho(x[1:9], p, t);
# 	dxBdt = bioreactor_julia_cho_cell(t, x[1:9], theta_M, theta_P, [u_F[1], u_F[2] + u_F[3]], u_Cin[1:2]);
	dxHdt = harvest_tank(t, x[10:13], [u_F[2], u_F[4]], x[6:8]);
	u_FC = u_F[4] + u_F[5] + u_F[6];
	u_Cin = [x[10:12]*u_F[4]; u_Cin[3]*u_F[6]]/u_FC;
	u_Cin[isnan.(u_Cin)].=0;
	dxCdt = Column_Rotavirus(t, x[14:end],theta_C,[u_FC;u_F[7]],u_Cin);
	dxdt=[dxBdt[:];dxHdt[:];dxCdt[:]];
	reshape(dxdt, length(dxdt), 1)
end
