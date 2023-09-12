include(joinpath(file_dir, "models/dynamic_test_data.jl"))
include(joinpath(file_dir, "models/device_models.jl"))

function build_system(sys_size)

    if isfile(joinpath(file_dir, "models/json_files/14Bus/$sys_size$Bus.json"))
        try
            sys = System(joinpath(file_dir, "models/json_files/14Bus/$sys_size$Bus.json"), runchecks = false)
        catch e
            @error e
            sys = System(joinpath(file_dir, "models/psse_files/$sys_size$Bus.raw"))#, joinpath(file_dir, "dyn_data.dyr"))
        end
    else

        sys = System("src/models/psse_files/$sys_size$Bus.raw")
        set_units_base_system!(sys, "DEVICE_BASE")

        df = solve_powerflow(sys)
        total_power=sum(df["bus_results"].P_gen)
        
        Gf=0.15
        GF=0.02
        trip_gen=0.04
        syncGen = collect(get_components(Generator, sys));
        trip_cap=total_power*trip_gen/0.7
        for g in syncGen
            if g.bus.number == 3
                set_base_power!(g, trip_cap)
            end
            if get_base_power(g) == 500.000
                set_base_power!(g, 200.00)
            elseif get_base_power(g) == 250.000
                set_base_power!(g, 175.00)
            end
        end
        
        bus_capacity = Dict()
        for g in syncGen
            bus_capacity[g.bus.name] = get_base_power(g)
        end
        
        total_capacity=sum(values(bus_capacity))

        active_pu = ((1-trip_gen)*total_power)/(total_capacity-trip_cap)

        for gen in syncGen
            if gen.bus.number != 3
                set_active_power!(gen, active_pu)
            end 
            H = H_min + rand(1)[1]
            D = D_min + 0.5*rand(1)[1]
            case_gen = dyn_gen_second_order(gen, H, D)
            add_component!(sys, case_gen, gen)
        end
        
        trip_capPU=trip_cap/total_capacity
        for g in syncGen
            if g.bus.number == 3
                set_base_power!(g, trip_cap)
                set_base_power!(g.dynamic_injector, trip_cap)
                set_active_power!(g, 0.7)
            elseif g.bus.number != 1 && g.bus.number != 3
                set_base_power!(g, bus_capacity[g.bus.name]*(1-GF-Gf))
                set_base_power!(g.dynamic_injector, bus_capacity[g.bus.name]*(1-GF-Gf))
            end
        
            if g.bus.number != 1
                storage=add_battery(sys, join(["GF_Battery-", g.bus.number]), g.bus.name, GF*bus_capacity[g.bus.name], get_active_power(g), get_reactive_power(g))
                add_component!(sys, storage)
                inverter=add_grid_forming(storage, GF*bus_capacity[g.bus.name])
                add_component!(sys, inverter, storage)
        
                storage=add_battery(sys, join(["Gf_Battery-", g.bus.number]), g.bus.name, Gf*bus_capacity[g.bus.name], get_active_power(g), get_reactive_power(g))
                add_component!(sys, storage)
                inverter=add_grid_following(storage, Gf*bus_capacity[g.bus.name])
                add_component!(sys, inverter, storage)
            end
        
        end 

        if !isfile(joinpath(file_dir, "models/json_files/$sys_size$Bus/$sys_size$Bus.json"))
            to_json(sys, joinpath(file_dir, "models/json_files/$sys_size$Bus/$sys_size$Bus.json"))
        end
    end
    return sys
end

function change_ibr_penetration!(sys, GF, Gf, ibr_bus, bus_capacity)
    
    set_units_base_system!(sys, "DEVICE_BASE")
    generators = [g for g in get_components(Generator, sys)]
    
    replace_gens = [g for g in generators if g.bus.number in ibr_bus]
    
    for g in replace_gens
        if occursin("Trip", g.name)==false
            gen = get_component(ThermalStandard, sys, g.name)
            set_base_power!(gen, bus_capacity[g.bus.number]*(1-GF-Gf))
            set_base_power!(gen.dynamic_injector, bus_capacity[g.bus.number]*(1-GF-Gf))

            gen = get_component(GenericBattery, sys, join(["GF_Battery-", g.bus.number]))
            set_base_power!(gen, bus_capacity[g.bus.number]*GF)
            set_base_power!(gen.dynamic_injector, bus_capacity[g.bus.number]*GF)
                
            gen = get_component(GenericBattery, sys, join(["Gf_Battery-", g.bus.number]))
            set_base_power!(gen, bus_capacity[g.bus.number]*Gf)
            set_base_power!(gen.dynamic_injector, bus_capacity[g.bus.number]*Gf)
        end
    end

end

function change_zone_ibr_penetration!(sys, GF, Gf, ibr_bus, bus_capacity)
    
    set_units_base_system!(sys, "DEVICE_BASE")
    generators = [g for g in get_components(Generator, sys)]
    
    replace_gens = [g for g in generators if g.bus.number in ibr_bus]
    
    for g in replace_gens
        if occursin("Trip", g.name)==false
            gen = get_component(ThermalStandard, sys, g.name)
            set_base_power!(gen, bus_capacity[g.bus.number]*(1-GF-Gf))
            set_base_power!(gen.dynamic_injector, bus_capacity[g.bus.number]*(1-GF-Gf))

            gen = get_component(GenericBattery, sys, join(["GF_Battery-", g.bus.number]))
            set_base_power!(gen, bus_capacity[g.bus.number]*GF)
            set_base_power!(gen.dynamic_injector, bus_capacity[g.bus.number]*GF)
                
            gen = get_component(GenericBattery, sys, join(["Gf_Battery-", g.bus.number]))
            set_base_power!(gen, bus_capacity[g.bus.number]*Gf)
            set_base_power!(gen.dynamic_injector, bus_capacity[g.bus.number]*Gf)
        end
    end

end