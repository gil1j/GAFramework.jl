### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ dde63eba-7ffd-11eb-00f1-0d7dbb9841f2
module GAFramework

import Base.@kwdef

using Random

export GAOptions, myGA



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
			if pop[i].fitness == 0
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
				if i == length(parents)
					p2 = parents[i-1]
				else
					p2 = parents[i+1]
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
					
# Library of crossover / mutation / selection functions
					
# Crossover

"One Point Crossover"
function OnePt_CX(p1,p2)
	if length(p1)>1 && length(p2)>1
		marker1 = rand(1:min(length(p1)-1,length(p2)-1))


		c1 = p1[1:marker1]*p2[marker1+1:end]
		c2 = p2[1:marker1]*p1[marker1+1:end]
	else
		c1 = p1
		c2 = p2
	end

	return c1,c2
end
					
function KB_CX(p1,p2)
	if length(p1)>1 && length(p2)>1
	
		marker1 = rand(1:length(p1)-1)

		c1 = p1[1:marker1]*join(rand(['>','<','+','-','.',',','[',']'],length(p1[marker1+1:end])))
		c2 = join(rand(['>','<','+','-','.',',','[',']'],length(p1[1:marker1])))*p1[marker1+1:end]
	
	else
		c1 = p1
		c2 = p2
	end
	
	return c1,c2
end

# Mutation
					
"mutation function as implemented by Kory Becker in her AI-programmer.
4 equiprobable mutations : delete, insert, modify or shift"
function mut(str)
	str_mut = collect(str)
	r = rand()

	if r<=0.25 && length(str_mut) > 1 #delete
		i = rand(1:length(str_mut))
		deleteat!(str_mut,i)
	elseif r<=0.5 #insert
		i = rand(1:length(str_mut)+1)
		insert!(str_mut,i,rand(['>','<','+','-','.',',','[',']']))
	elseif r<=0.75 #modify
		i = rand(1:length(str))
		str_mut[i] = rand(['>','<','+','-','.',',','[',']'])
	else #shift
		if rand()<0.5
			str_mut = circshift(str_mut,1)
		else
			str_mut = circshift(str_mut,-1)
		end
	end

	return join(str_mut)
end
					
# Selection
					
begin
	using StatsBase
	
	"roulette selection, probability of an individual being selected is proportional to its fitness. Note : roulette selection kills strict elitism"
	function roulette(fitPop, N)
		fitPopRel = abs.(fitPop.-maximum(fitPop))
		fitPopRel = fitPopRel .+ 0.001
		prob = Weights(fitPopRel./sum(fitPopRel))
		selectionFit = sample(fitPopRel,prob,N,replace=false)

		selectionInd = []
		for i in 1:length(selectionFit)
			append!(selectionInd,[findfirst(fitPopRel.==selectionFit[i])])
			fitPopRel[findfirst(fitPopRel.==selectionFit[i])] = -1
		end

		return selectionInd
	end
end
					
"trivial selection"
function mySelection(fitPop,popSize)
	indexSorted = sortperm(fitPop)
	toKeep = indexSorted[1:popSize]
	
	return toKeep
end
					
end

# ╔═╡ Cell order:
# ╠═dde63eba-7ffd-11eb-00f1-0d7dbb9841f2
