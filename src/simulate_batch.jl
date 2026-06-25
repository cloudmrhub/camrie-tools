if any(==("--gpu"),ARGS)
	try
		@static if Sys.isapple()
			@eval using Metal
		else
			@eval using CUDA
		end
	catch
	end
end
using KomaInterface
KomaInterface.main()