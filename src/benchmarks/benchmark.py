
class Benchmark:
  def Benchmark(self, func):
    self.f = func
  
  def run(self, args):
    return self.f(args)
  
  def __str__(self):
    return "benchmark"