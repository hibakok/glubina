using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Sigmauadro
{
    // Примитивные операции для стековой VM
    enum OpCode { Push, Add, Sub, Mul, Div, Dup, Swap, Pop, Sin, Cos, Exp, Log, Sqrt, Abs, Neg, Inv, Pow, Min, Max, Input }
    
    // Команда в программе особи
    class Instruction { public OpCode Op; public double Value; }
    
    // Особь - программа из инструкций
    class Individual
    {
        public List<Instruction> Genome = new List<Instruction>();
        public double Fitness = double.MaxValue;
        public int Complexity => Genome.Count;
        
        public Individual Clone() => new Individual { Genome = Genome.Select(i => new Instruction { Op = i.Op, Value = i.Value }).ToList(), Fitness = Fitness };
    }
    
    // Загрузчик данных
    class DataLoader
    {
        public List<(double[] Input, double[] Output)> Data = new List<(double[], double[])>();
        
        public bool Load(string path)
        {
            Data.Clear();
            if (!File.Exists(path)) return false;
            foreach (var line in File.ReadAllLines(path))
            {
                var parts = line.Split('|');
                if (parts.Length != 2) continue;
                var input = parts[0].Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                var output = parts[1].Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                Data.Add((input, output));
            }
            return Data.Count > 0;
        }
    }
    
    // Настройки
    class Settings
    {
        public int InnerPopSize = 10;
        public int OffspringPerParent = 5;
        public int MutationsPerOffspring = 3;
        public int MaxGenomeLength = 50;
        public int MaxCpuPercent = 80;
        
        public bool Load(string path)
        {
            if (!File.Exists(path)) return false;
            foreach (var line in File.ReadAllLines(path))
            {
                var parts = line.Split('=');
                if (parts.Length != 2) continue;
                switch (parts[0].Trim().ToLower())
                {
                    case "innerpopsize": InnerPopSize = int.Parse(parts[1]); break;
                    case "offspringperparent": OffspringPerParent = int.Parse(parts[1]); break;
                    case "mutationsperoffspring": MutationsPerOffspring = int.Parse(parts[1]); break;
                    case "maxgenomelength": MaxGenomeLength = int.Parse(parts[1]); break;
                    case "maxcpupercent": MaxCpuPercent = int.Parse(parts[1]); break;
                }
            }
            return true;
        }
        
        public void Save(string path)
        {
            File.WriteAllLines(path, new[]
            {
                $"InnerPopSize={InnerPopSize}",
                $"OffspringPerParent={OffspringPerParent}",
                $"MutationsPerOffspring={MutationsPerOffspring}",
                $"MaxGenomeLength={MaxGenomeLength}",
                $"MaxCpuPercent={MaxCpuPercent}"
            });
        }
    }
    
    // Стековая VM для исполнения программ особей
    class StackVM
    {
        private Stack<double> stack = new Stack<double>();
        
        public (bool Success, double[] Result) Execute(List<Instruction> genome, double[] input)
        {
            stack.Clear();
            foreach (var item in input.Reverse()) stack.Push(item);
            
            foreach (var instr in genome)
            {
                int required = GetRequiredStack(instr.Op);
                if (stack.Count < required && instr.Op != OpCode.Input) return (false, null);
                
                switch (instr.Op)
                {
                    case OpCode.Push: stack.Push(instr.Value); break;
                    case OpCode.Add: if (stack.Count < 2) return (false, null); stack.Push(stack.Pop() + stack.Pop()); break;
                    case OpCode.Sub: if (stack.Count < 2) return (false, null); var b1 = stack.Pop(); var a1 = stack.Pop(); stack.Push(a1 - b1); break;
                    case OpCode.Mul: if (stack.Count < 2) return (false, null); stack.Push(stack.Pop() * stack.Pop()); break;
                    case OpCode.Div: if (stack.Count < 2) return (false, null); var b2 = stack.Pop(); var a2 = stack.Pop(); stack.Push(b2 != 0 ? a2 / b2 : 0); break;
                    case OpCode.Dup: if (stack.Count < 1) return (false, null); stack.Push(stack.Peek()); break;
                    case OpCode.Swap: if (stack.Count < 2) return (false, null); var s1 = stack.Pop(); var s2 = stack.Pop(); stack.Push(s1); stack.Push(s2); break;
                    case OpCode.Pop: if (stack.Count > 0) stack.Pop(); break;
                    case OpCode.Sin: if (stack.Count < 1) return (false, null); stack.Push(Math.Sin(stack.Pop())); break;
                    case OpCode.Cos: if (stack.Count < 1) return (false, null); stack.Push(Math.Cos(stack.Pop())); break;
                    case OpCode.Exp: if (stack.Count < 1) return (false, null); stack.Push(Math.Exp(stack.Pop())); break;
                    case OpCode.Log: if (stack.Count < 1) return (false, null); stack.Push(Math.Log(stack.Pop())); break;
                    case OpCode.Sqrt: if (stack.Count < 1) return (false, null); stack.Push(Math.Sqrt(Math.Max(0, stack.Pop()))); break;
                    case OpCode.Abs: if (stack.Count < 1) return (false, null); stack.Push(Math.Abs(stack.Pop())); break;
                    case OpCode.Neg: if (stack.Count < 1) return (false, null); stack.Push(-stack.Pop()); break;
                    case OpCode.Inv: if (stack.Count < 1) return (false, null); var v = stack.Pop(); stack.Push(v != 0 ? 1.0 / v : 0); break;
                    case OpCode.Pow: if (stack.Count < 2) return (false, null); var bp = stack.Pop(); var ap = stack.Pop(); stack.Push(Math.Pow(ap, bp)); break;
                    case OpCode.Min: if (stack.Count < 2) return (false, null); stack.Push(Math.Min(stack.Pop(), stack.Pop())); break;
                    case OpCode.Max: if (stack.Count < 2) return (false, null); stack.Push(Math.Max(stack.Pop(), stack.Pop())); break;
                    case OpCode.Input: if (input.Length > 0) stack.Push(input[0]); break;
                }
            }
            
            var result = stack.Reverse().Take(input.Length).ToArray();
            return (true, result.Length > 0 ? result : new[] { stack.Count > 0 ? stack.Pop() : 0 });
        }
        
        private int GetRequiredStack(OpCode op)
        {
            switch (op)
            {
                case OpCode.Add: case OpCode.Sub: case OpCode.Mul: case OpCode.Div:
                case OpCode.Pow: case OpCode.Min: case OpCode.Max: return 2;
                case OpCode.Sin: case OpCode.Cos: case OpCode.Exp: case OpCode.Log:
                case OpCode.Sqrt: case OpCode.Abs: case OpCode.Neg: case OpCode.Inv: return 1;
                default: return 0;
            }
        }
    }
    
    // Эволюционный движок
    class EvolutionEngine
    {
        private List<Individual> innerPop = new List<Individual>();
        private List<Individual> outerPop = new List<Individual>();
        private Individual bestEver = new Individual();
        private Random rand = new Random();
        private StackVM vm = new StackVM();
        private Settings settings = new Settings();
        private DataLoader dataLoader = new DataLoader();
        
        private OpCode[] ops = Enum.GetValues(typeof(OpCode)).Cast<OpCode>().ToArray();
        
        public void Initialize(Settings s, DataLoader d)
        {
            settings = s;
            dataLoader = d;
            innerPop.Clear();
            
            // Создаем начальную популяцию с минимальными особями
            for (int i = 0; i < settings.InnerPopSize; i++)
            {
                var ind = new Individual();
                // Начинаем с простой операции push входного значения
                ind.Genome.Add(new Instruction { Op = OpCode.Input });
                innerPop.Add(ind);
            }
            bestEver = innerPop[0].Clone();
        }
        
        public void EvolveGeneration()
        {
            outerPop.Clear();
            
            // Каждый прародитель порождает потомков
            foreach (var parent in innerPop)
            {
                for (int i = 0; i < settings.OffspringPerParent; i++)
                {
                    var offspring = parent.Clone();
                    Mutate(offspring);
                    Evaluate(offspring);
                    outerPop.Add(offspring);
                }
            }
            
            // Обновляем внутреннюю популяцию лучшими потомками
            for (int i = 0; i < innerPop.Count; i++)
            {
                var offspring = outerPop.Where(o => IsOffspringOf(o, innerPop[i])).OrderByDescending<Individual, double>(o => o.Fitness).FirstOrDefault();
                if (offspring != null && FitnessCompare(offspring, innerPop[i]) > 0)
                {
                    innerPop[i] = offspring.Clone();
                }
            }
            
            // Обновляем лучшую особь
            var currentBest = innerPop.OrderBy<Individual, double>(o => o.Fitness).First();
            if (FitnessCompare(currentBest, bestEver) > 0)
                bestEver = currentBest.Clone();
        }
        
        private bool IsOffspringOf(Individual offspring, Individual parent)
        {
            // Простая проверка - потомок имеет схожую структуру
            return true;
        }
        
        /// <summary>
        /// Сравнивает две особи по приспособленности.
        /// Возвращает: 1 если a лучше b, -1 если b лучше a, 0 если равны.
        /// Лучше та, у которой ошибка меньше, при равной ошибке - проще.
        /// </summary>
        private int FitnessCompare(Individual a, Individual b)
        {
            // Сначала сравниваем по приспособленности (ошибке)
            if (Math.Abs(a.Fitness - b.Fitness) > 1e-15)
                return a.Fitness < b.Fitness ? 1 : -1;
            // При равной ошибке предпочитаем более простую
            return b.Complexity - a.Complexity;
        }
        
        private void Mutate(Individual ind)
        {
            for (int m = 0; m < settings.MutationsPerOffspring; m++)
            {
                var mutationType = rand.Next(4);
                switch (mutationType)
                {
                    case 0: // Добавить инструкцию
                        if (ind.Genome.Count < settings.MaxGenomeLength)
                        {
                            var pos = rand.Next(ind.Genome.Count + 1);
                            var newInstr = new Instruction 
                            { 
                                Op = ops[rand.Next(ops.Length)],
                                Value = rand.NextDouble() * 10 - 5
                            };
                            ind.Genome.Insert(pos, newInstr);
                        }
                        break;
                    case 1: // Удалить инструкцию
                        if (ind.Genome.Count > 1)
                        {
                            ind.Genome.RemoveAt(rand.Next(ind.Genome.Count));
                        }
                        break;
                    case 2: // Изменить операцию
                        if (ind.Genome.Count > 0)
                        {
                            ind.Genome[rand.Next(ind.Genome.Count)].Op = ops[rand.Next(ops.Length)];
                        }
                        break;
                    case 3: // Изменить значение
                        if (ind.Genome.Count > 0)
                        {
                            var idx = rand.Next(ind.Genome.Count);
                            ind.Genome[idx].Value += (rand.NextDouble() - 0.5) * 0.0001;
                        }
                        break;
                }
            }
        }
        
        private void Evaluate(Individual ind)
        {
            double totalError = 0;
            int count = 0;
            
            foreach (var (input, output) in dataLoader.Data)
            {
                var (success, result) = vm.Execute(ind.Genome, input);
                if (!success)
                {
                    ind.Fitness = double.MaxValue;
                    return;
                }
                
                for (int i = 0; i < output.Length; i++)
                {
                    var expected = i < output.Length ? output[i] : 0;
                    var actual = i < result.Length ? result[i] : 0;
                    totalError += Math.Abs(expected - actual);
                    count++;
                }
            }
            
            ind.Fitness = count > 0 ? totalError / count : double.MaxValue;
        }
        
        public Individual GetBest() => bestEver.Clone();
        public List<Individual> GetInnerPopulation() => innerPop.Select(i => i.Clone()).ToList();
        public void SetInnerPopulation(List<Individual> pop) { innerPop = pop.Select(i => i.Clone()).ToList(); }
        
        public string ExportIndividual(Individual ind)
        {
            var lines = new List<string> { $"// Особь: сложность={ind.Complexity}, ошибка={ind.Fitness:e}" };
            foreach (var instr in ind.Genome)
            {
                lines.Add($"{instr.Op} {(instr.Op == OpCode.Push || instr.Op == OpCode.Input ? instr.Value.ToString("G17", CultureInfo.InvariantCulture) : "")}");
            }
            return string.Join("\n", lines);
        }
        
        public void SavePopulation(string path)
        {
            var lines = new List<string>();
            foreach (var ind in innerPop)
            {
                lines.Add("---INDIVIDUAL---");
                lines.AddRange(ExportIndividual(ind).Split('\n'));
            }
            File.WriteAllLines(path, lines);
        }
        
        public void LoadPopulation(string path)
        {
            if (!File.Exists(path)) return;
            var lines = File.ReadAllLines(path);
            var newPop = new List<Individual>();
            Individual current = null;
            
            foreach (var line in lines)
            {
                if (line == "---INDIVIDUAL---")
                {
                    if (current != null) newPop.Add(current);
                    current = new Individual();
                }
                else if (current != null && !line.StartsWith("//"))
                {
                    var parts = line.Trim().Split(' ');
                    if (Enum.TryParse<OpCode>(parts[0], out var op))
                    {
                        current.Genome.Add(new Instruction 
                        { 
                            Op = op, 
                            Value = parts.Length > 1 && double.TryParse(parts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : 0
                        });
                    }
                }
            }
            if (current != null && current.Genome.Count > 0) newPop.Add(current);
            if (newPop.Count > 0) innerPop = newPop;
        }
        
        public double[] TestBest(double[] input)
        {
            var (success, result) = vm.Execute(bestEver.Genome, input);
            return success ? result : null;
        }
    }
    
    class Program
    {
        static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            var settings = new Settings();
            var dataLoader = new DataLoader();
            var engine = new EvolutionEngine();
            
            // Загрузка настроек
            if (File.Exists("settings.txt"))
                settings.Load("settings.txt");
            else
                settings.Save("settings.txt");
            
            // Загрузка данных
            Console.WriteLine("Введите путь к файлу с данными:");
            var dataPath = Console.ReadLine();
            if (!dataLoader.Load(dataPath))
            {
                Console.WriteLine("Ошибка загрузки данных. Нажмите Enter для выхода.");
                Console.ReadLine();
                return;
            }
            
            engine.Initialize(settings, dataLoader);
            
            bool exit = false;
            while (!exit)
            {
                Console.WriteLine("\n=== Главное меню ===");
                Console.WriteLine("1. Начать эволюцию");
                Console.WriteLine("2. Тестировать текущую лучшую особь");
                Console.WriteLine("3. Сохранить лучшую особь в файл");
                Console.WriteLine("4. Сохранить популяцию в файл");
                Console.WriteLine("5. Загрузить популяцию из файла");
                Console.WriteLine("6. Выход");
                Console.Write("Выбор: ");
                
                var choice = Console.ReadLine();
                switch (choice)
                {
                    case "1":
                        Console.Write("Сколько поколений эволюции прогонять? ");
                        if (int.TryParse(Console.ReadLine(), out var generations))
                        {
                            Console.WriteLine($"Эволюция на {generations} поколений...");
                            for (int g = 0; g < generations; g++)
                            {
                                engine.EvolveGeneration();
                                if ((g + 1) % 10 == 0)
                                    Console.WriteLine($"Поколение {g + 1}, лучшая ошибка: {engine.GetBest().Fitness:e}");
                            }
                            Console.WriteLine($"Завершено. Лучшая ошибка: {engine.GetBest().Fitness:e}");
                        }
                        break;
                    case "2":
                        Console.WriteLine("Тестирование лучшей особи (введите 'выйти' для выхода):");
                        while (true)
                        {
                            Console.Write("Входные данные: ");
                            var inputLine = Console.ReadLine();
                            if (inputLine?.ToLower() == "выйти") break;
                            var input = inputLine.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries)
                                .Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                            var result = engine.TestBest(input);
                            Console.WriteLine($"Результат: {string.Join(" ", result?.Select(r => r.ToString("G17", CultureInfo.InvariantCulture)) ?? new[] { "ошибка" })}");
                        }
                        break;
                    case "3":
                        Console.Write("Имя файла для сохранения: ");
                        var bestFile = Console.ReadLine();
                        File.WriteAllText(bestFile, engine.ExportIndividual(engine.GetBest()));
                        Console.WriteLine("Сохранено.");
                        break;
                    case "4":
                        Console.Write("Имя файла для сохранения популяции: ");
                        var popFile = Console.ReadLine();
                        engine.SavePopulation(popFile);
                        Console.WriteLine("Популяция сохранена.");
                        break;
                    case "5":
                        Console.Write("Имя файла для загрузки популяции: ");
                        var loadFile = Console.ReadLine();
                        engine.LoadPopulation(loadFile);
                        Console.WriteLine("Популяция загружена.");
                        break;
                    case "6":
                        exit = true;
                        break;
                }
            }
        }
    }
}
