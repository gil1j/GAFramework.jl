### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ dde63eba-7ffd-11eb-00f1-0d7dbb9841f2
module GAFramework

import Base.@kwdef

export BFProg, GAOptions, myGA

"struct for the usage of brainfuck programs in a genetic algorithm, storing the program and its fitness"

mutable struct BFProg
	program
	fitness::Int64
	
	function BFProg(maxProgSize::Int64)
		return new(generate_rand_prog(maxProgSize),10^10)
	end
	
	function BFProg(program,fitness::Int64)
		return new(program,fitness)
	end
end

"struct for the usage of myGA(), containing all the necessary options"

@kwdef mutable struct GAOptions
	popSize::Int64=100
	maxProgSize::Int64=500
	crossoverRate::Float64=0.8
	mutationRate::Float64=0.1
	showEvery::Int64=100
	targetFit::Int64=0
	maxGen::Int64=1000000
	progTicksLim::Int64=5000
	elitism::Float64=0.1
end

begin
	using TickTock
	
	"framework for Genetic Algorithms, developed primarily for usage with Brainfuck programs as individuals"
	
	function myGA(generator,fitness,crossover,selection,mutation,options)
		tick()
		
		pop = [generator(options.maxProgSize) for i in 1:options.popSize]
		
		Threads.@threads for i in 1:length(pop)
			if pop[i].fitness == 10^10
				pop[i].fitness = fitness(pop[i].program,options.progTicksLim)
			end
		end
		
		stop = false
		
		gen = 0
		
		while(!stop)
			gen += 1
			
			parents = shuffle(pop)
			
			childs = []
			
			#crossover
			
			for i in 1:2:length(parents)

				p1 = parents[i]
				if options.popSize % 2 == 0
					p2 = parents[i+1]
				else
					p2 = parents[i-1]
				end
				
				if rand() <= options.crossoverRate
	
					c1_prog,c2_prog = crossover(p1.program,p2.program)
					c1 = generator(c1_prog,10^10)
					c2 = generator(c2_prog,10^10)
	
					append!(childs,[c1,c2])
				else
					c1 = p1
					c2 = p2
					
					append!(childs,[c1,c2])
				end
			end
			
			#mutation
			
			for i in 1:length(childs)
				
				if rand() <= options.mutationRate
					mut_prog = mutation(childs[i].program)
					mut = generator(mut_prog,10^10)
					
					childs[i] = mut
				end
			end
			
			if length(childs)<options.popSize
				append!(childs,[generator(options.maxProgSize) for i in 1:options.popSize-length(childs)])
			end
			
			#fitness
			
			Threads.@threads for i in 1:length(childs)
				if childs[i].fitness == 10^10
					childs[i].fitness = fitness(childs[i].program,options.progTicksLim)
				end
			end
			
			fitParents = Array{Int64,1}(undef,length(parents))
			
			for i in 1:length(parents)
				fitParents[i] = parents[i].fitness
			end
			
			indexElite = selection(fitParents,Int(round(length(parents)*options.elitism)))
			
			append!(childs,parents[indexElite])
			
			fitChilds = Array{Int64,1}(undef,length(childs))
			
			for i in 1:length(childs)
				fitChilds[i] = childs[i].fitness
			end
			
			#data
			
			bestFit = minimum(fitChilds)
			bestInd = childs[findfirst(fitChilds.==bestFit)].program
			elapsedTime = peektimer()
			
			if gen % options.showEvery == 0
				@show bestFit,bestInd,elapsedTime,gen
			end
			
			#selection
			
			indexToKeep = selection(fitChilds,options.popSize)
			indexToDelete = [x for x ∈ 1:length(childs) if x ∉ indexToKeep]
			
			deleteat!(childs,indexToDelete)
			pop = childs
			
			#stop criterion
			
			if gen > options.maxGen
				stop = true
				@show "Not good enough ..."
				return bestFit,bestInd,elapsedTime,gen
			elseif bestFit == options.targetFit
				stop = true
				@show "Success !"
				return bestFit,bestInd,elapsedTime,gen
			end
		end
	end
end

end

# ╔═╡ Cell order:
# ╠═dde63eba-7ffd-11eb-00f1-0d7dbb9841f2
