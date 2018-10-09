using Statistics
using Gadfly
using ProgressMeter

mutable struct IsingModel
    N :: Int64
    J :: Float64
    k :: Float64
    T :: Float64
    spins :: Array{Int8, 2}
    energy :: Float64
    magnetization :: Float64
end

function Base.copy(model :: IsingModel)
    return IsingModel(model.N, model.J, model.k, model.T, copy(model.spins),
                      model.energy, model.magnetization)
end

function update!(model :: IsingModel)
    function update_magnetization!(model :: IsingModel)
        N₊ = count(i -> i == 1, model.spins)
        N₋ = count(i -> i == -1, model.spins)
        model.magnetization = (N₊ - N₋) / model.N^2
    end

    function update_energy!(model :: IsingModel)
        # periodic boundary condition
        energy = 0.0
        for x in 1:model.N
            for y in 1:model.N
                if (x == model.N) && (y == model.N)
                    energy += model.spins[x, y] * (model.spins[x, 1] + model.spins[1, y])
                elseif (x == model.N) && (y != model.N)
                    energy += model.spins[x, y] * (model.spins[x, y + 1] + model.spins[1, y])
                elseif (x != model.N) && (y == model.N)
                    energy += model.spins[x, y] * (model.spins[x, 1] + model.spins[x + 1, y])
                else
                    energy += model.spins[x, y] * (model.spins[x, y + 1] + model.spins[x + 1, y])
                end
            end
        end

        model.energy = -model.J * energy

        return nothing
    end

    update_magnetization!(model)
    update_energy!(model)

    return nothing
end

function init_model(N :: Int64, J :: Float64, k :: Float64, T :: Float64)
    if N <= 2
        error("N must be greater than 2")
    end

    model = IsingModel(N, J, k, T, rand([-1, 1], (N, N)), 0.0, 0.0)
    update!(model)

    return model
end

function metropolis(model :: IsingModel)
    candidate_model = copy(model)

    change_x, change_y = rand(1:model.N), rand(1:model.N)
    candidate_model.spins[change_x, change_y] *= Int8(-1)
    update!(candidate_model)

    if candidate_model.energy <= model.energy
        model = copy(candidate_model)
    elseif (model.energy - candidate_model.energy) / (model.k * model.T) >= log(rand())
        model = copy(candidate_model)
    end

    return model
end

function ising_simulation(N :: Int64, J :: Float64, k :: Float64, T :: Float64;
                          niters = 10000 :: Int64, burnin = 5000 :: Int64)
    magnetizations = Array{Float64}(undef, niters)
    energies = Array{Float64}(undef, niters)

    model = init_model(N, J, k, T)

    for i in 1:niters
        magnetizations[i] = model.magnetization
        energies[i] = model.energy
        model = metropolis(model)
    end

    return mean(magnetizations[burnin + 1:niters]), mean(energies[burnin + 1:niters])

    #theme = Theme(background_color = "white", line_width = 2px)
    #p = plot(x = 1:niters, y = magnetizations, theme, Geom.line,
    #         Guide.xlabel("iterations"), Guide.ylabel("magnetizations"), Guide.title("magnetizations(kT = $(k * T))"))
    #img = SVG("magnetizations_kT$(k*T).svg", 960px, 540px)
    #draw(img, p)

    #p = plot(x = 1:niters, y = energies, theme, Geom.line,
    #         Guide.xlabel("iterations"), Guide.ylabel("energies"), Guide.title("magnetizations(kT = $(k * T))"))
    #img = SVG("energies_kT$(k*T).svg", 960px, 540px)
    #draw(img, p)
end

function run_simulation()
    temperatures = range(0.001, step = 0.01, stop = 10.0)

    magnetizations = Array{Float64}(undef, length(temperatures))
    energies = Array{Float64}(undef, length(temperatures))

    progress = Progress(length(temperatures))
    for i in 1:length(temperatures)
        magnetizations[i], energies[i] = ising_simulation(20, 1.0, 1.0,
                                                          temperatures[i],
                                                          niters = 100000,
                                                          burnin = 50000)
        next!(progress)
    end

    theme = Theme(background_color = "white", line_width = 2px)
    p = plot(x = temperatures, y = abs.(magnetizations), theme, Geom.line,
             Guide.xlabel("kT"), Guide.ylabel("|magnetizations|"), Guide.title("abs magnetizations"))
    img = SVG("magnetizations_vs_kT.svg", 960px, 540px)
    draw(img, p)

    p = plot(x = temperatures, y = energies, theme, Geom.line,
             Guide.xlabel("kT"), Guide.ylabel("energies"), Guide.title("energies"))
    img = SVG("energies_vs_kT.svg", 960px, 540px)
    draw(img, p)
end

run_simulation()
