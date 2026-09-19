using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Threading.Tasks;
using System.Diagnostics;

namespace Sigmauadro
{
    // Операции стековой VM (Тьюринг-полная)
    enum OpCode { Push, Add, Sub, Mul, Div, Dup, Swap, Pop, Sin, Cos, Exp, Log, Sqrt, Abs, Neg, Inv, Pow, Min, Max, Input, Jump, Branch, Eq, Lt, Gt }
    
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
    
    // VM со стеком и регистрами для Тьюринг-полноты
    class StackVM
    {
        private Stack<double> stack = new Stack<double>();
        private int ip = 0; // instruction pointer
        
        public (bool Success, double[] Result) Execute(List<Instruction> genome, double[] input)
        {
            stack.Clear(); ip = 0;
            foreach (var item in input.Reverse()) stack.Push(item);
            
            while (ip < genome.Count)
            {
                var instr = genome[ip];
                int required = GetRequiredStack(instr.Op);
                if (stack.Count < required && instr.Op != OpCode.Input) return (false, null);
                
                switch (instr.Op)
                {
                    case OpCode.Push: stack.Push(instr.Value); break;
                    case OpCode.Add: var ba = stack.Pop(); var aa = stack.Pop(); stack.Push(aa + ba); break;
                    case OpCode.Sub: var bb = stack.Pop(); var ab = stack.Pop(); stack.Push(ab - bb); break;
                    case OpCode.Mul: var bm = stack.Pop(); var am = stack.Pop(); stack.Push(am * bm); break;
                    case OpCode.Div: var bd = stack.Pop(); var ad = stack.Pop(); stack.Push(bd != 0 ? ad / bd : 0); break;
                    case OpCode.Dup: stack.Push(stack.Peek()); break;
                    case OpCode.Swap: var x = stack.Pop(); var y = stack.Pop(); stack.Push(x); stack.Push(y); break;
                    case OpCode.Pop: if (stack.Count > 0) stack.Pop(); break;
                    case OpCode.Sin: stack.Push(Math.Sin(stack.Pop())); break;
                    case OpCode.Cos: stack.Push(Math.Cos(stack.Pop())); break;
                    case OpCode.Exp: stack.Push(Math.Exp(stack.Pop())); break;
                    case OpCode.Log: stack.Push(Math.Log(stack.Pop())); break;
                    case OpCode.Sqrt: stack.Push(Math.Sqrt(Math.Max(0, stack.Pop()))); break;
                    case OpCode.Abs: stack.Push(Math.Abs(stack.Pop())); break;
                    case OpCode.Neg: stack.Push(-stack.Pop()); break;
                    case OpCode.Inv: var vi = stack.Pop(); stack.Push(vi != 0 ? 1.0 / vi : 0); break;
                    case OpCode.Pow: var bp = stack.Pop(); var ap = stack.Pop(); stack.Push(Math.Pow(ap, bp)); break;
                    case OpCode.Min: var bmin = stack.Pop(); var amin = stack.Pop(); stack.Push(Math.Min(amin, bmin)); break;
                    case OpCode.Max: var bmax = stack.Pop(); var amax = stack.Pop(); stack.Push(Math.Max(amax, bmax)); break;
                    case OpCode.Input: if (input.Length > 0) stack.Push(input[0]); break;
                    case OpCode.Eq: var be = stack.Pop(); var ae = stack.Pop(); stack.Push(Math.Abs(ae - be) < 1e-15 ? 1 : 0); break;
                    case OpCode.Lt: var bl = stack.Pop(); var al = stack.Pop(); stack.Push(al < bl ? 1 : 0); break;
                    case OpCode.Gt: var bg = stack.Pop(); var ag = stack.Pop(); stack.Push(ag > bg ? 1 : 0); break;
                    case OpCode.Branch: var cond = stack.Pop(); if (Math.Abs(cond) < 1e-15) ip = (int)Math.Abs(instr.Value); break;
                    case OpCode.Jump: ip = (int)Math.Abs(instr.Value); continue;
                }
                ip++;
            }
            
            // Создаем копию стека через ToArray для безопасного копирования
            var res = stack.ToArray().Reverse().ToArray();
            return (true, res.Length > 0 ? res : new double[] { 0 });
        }
        
        private int GetRequiredStack(OpCode op)
        {
            switch (op)
            {
                case OpCode.Add: case OpCode.Sub: case OpCode.Mul: case OpCode.Div:
                case OpCode.Pow: case OpCode.Min: case OpCode.Max: case OpCode.Eq:
                case OpCode.Lt: case OpCode.Gt: case OpCode.Swap: return 2;
                case OpCode.Sin: case OpCode.Cos: case OpCode.Exp: case OpCode.Log:
                case OpCode.Sqrt: case OpCode.Abs: case OpCode.Neg: case OpCode.Inv:
                case OpCode.Branch: return 1;
                default: return 0;
            }
        }
    }
    
    // Эволюционный движок с параллельной оценкой
    class EvolutionEngine
    {
        private List<Individual> innerPop = new List<Individual>();
        private Individual bestEver = new Individual();
        private Random rand = new Random();
        private StackVM vm = new StackVM();
        private Settings settings = new Settings();
        private DataLoader dataLoader = new DataLoader();
        private OpCode[] ops = Enum.GetValues(typeof(OpCode)).Cast<OpCode>().ToArray();
        
        public void Initialize(Settings s, DataLoader d)
        {
            settings = s; dataLoader = d; innerPop.Clear();
            // Начальная популяция - пустые особи (эволюция с нуля)
            for (int i = 0; i < settings.InnerPopSize; i++)
                innerPop.Add(new Individual());
            bestEver = new Individual();
        }
        
        public void EvolveGeneration()
        {
            var offspringList = new List<(int parentIdx, Individual ind)>();
            // Параллельное порождение потомков
            var tasks = new Task[innerPop.Count];
            for (int p = 0; p < innerPop.Count; p++)
            {
                int parentIdx = p;
                tasks[p] = Task.Run(() =>
                {
                    for (int i = 0; i < settings.OffspringPerParent; i++)
                    {
                        var offspring = innerPop[parentIdx].Clone();
                        Mutate(offspring);
                        Evaluate(offspring);
                        lock (offspringList) { offspringList.Add((parentIdx, offspring)); }
                    }
                });
            }
            Task.WaitAll(tasks);
            
            // Замена прародителей превосходящими потомками
            foreach (var group in offspringList.GroupBy(o => o.parentIdx).OrderBy(g => g.Min(x => x.ind.Fitness)))
            {
                var bestOffspring = group.OrderBy(x => x.ind.Fitness).First().ind;
                if (FitnessCompare(bestOffspring, innerPop[group.Key]) > 0)
                    innerPop[group.Key] = bestOffspring.Clone();
            }
            
            // Обновление лучшей особи
            var currentBest = innerPop.OrderBy(o => o.Fitness).First();
            if (FitnessCompare(currentBest, bestEver) > 0)
                bestEver = currentBest.Clone();
        }
        
        // Сравнение особей: 1 если a лучше, -1 если b лучше, 0 если равны
        // Лучше: меньше ошибка, при равенстве - проще
        private int FitnessCompare(Individual a, Individual b)
        {
            if (Math.Abs(a.Fitness - b.Fitness) > 1e-15)
                return a.Fitness < b.Fitness ? 1 : -1;
            return b.Complexity - a.Complexity;
        }
        
        // Мутация: добавление/удаление/изменение инструкции, изменение значения
        private void Mutate(Individual ind)
        {
            for (int m = 0; m < settings.MutationsPerOffspring; m++)
            {
                switch (rand.Next(4))
                {
                    case 0: // Добавить
                        if (ind.Genome.Count < settings.MaxGenomeLength)
                            ind.Genome.Insert(rand.Next(ind.Genome.Count + 1), new Instruction { Op = ops[rand.Next(ops.Length)], Value = rand.NextDouble() * 10 - 5 });
                        break;
                    case 1: // Удалить
                        if (ind.Genome.Count > 1)
                            ind.Genome.RemoveAt(rand.Next(ind.Genome.Count));
                        break;
                    case 2: // Изменить опкод
                        if (ind.Genome.Count > 0)
                            ind.Genome[rand.Next(ind.Genome.Count)].Op = ops[rand.Next(ops.Length)];
                        break;
                    case 3: // Изменить значение
                        if (ind.Genome.Count > 0)
                            ind.Genome[rand.Next(ind.Genome.Count)].Value += (rand.NextDouble() - 0.5) * 0.0001;
                        break;
                }
            }
        }
        
        // Оценка особи: средняя абсолютная ошибка на всех парах данных
        private void Evaluate(Individual ind)
        {
            double totalError = 0;
            int count = 0;
            foreach (var (input, output) in dataLoader.Data)
            {
                var (success, result) = vm.Execute(ind.Genome, input);
                if (!success) { ind.Fitness = double.MaxValue; return; }
                for (int i = 0; i < output.Length; i++)
                {
                    totalError += Math.Abs((i < output.Length ? output[i] : 0) - (i < result.Length ? result[i] : 0));
                    count++;
                }
            }
            ind.Fitness = count > 0 ? totalError / count : double.MaxValue;
        }
        
        public Individual GetBest() => bestEver.Clone();
        public List<Individual> GetInnerPopulation() => innerPop.Select(i => i.Clone()).ToList();
        public void SetInnerPopulation(List<Individual> pop) { innerPop = pop.Select(i => i.Clone()).ToList(); }
        
        // Экспорт особи в читаемый формат
        public string ExportIndividual(Individual ind)
        {
            var lines = new List<string> { $"// Сложность={ind.Complexity}, Ошибка={ind.Fitness:e}" };
            foreach (var i in ind.Genome)
                lines.Add($"{i.Op}{(i.Op == OpCode.Push || i.Op == OpCode.Input || i.Op == OpCode.Jump || i.Op == OpCode.Branch ? " " + i.Value.ToString("G17", CultureInfo.InvariantCulture) : "")}");
            return string.Join("\n", lines);
        }
        
        // Сохранение популяции в файл
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
        
        // Загрузка популяции из файла
        public void LoadPopulation(string path)
        {
            if (!File.Exists(path)) return;
            var newPop = new List<Individual>();
            Individual cur = null;
            foreach (var line in File.ReadAllLines(path))
            {
                if (line == "---INDIVIDUAL---") { if (cur != null) newPop.Add(cur); cur = new Individual(); }
                else if (cur != null && !line.StartsWith("//") && !string.IsNullOrWhiteSpace(line))
                {
                    var parts = line.Trim().Split(' ');
                    if (Enum.TryParse<OpCode>(parts[0], out var op))
                        cur.Genome.Add(new Instruction { Op = op, Value = parts.Length > 1 && double.TryParse(parts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out var v) ? v : 0 });
                }
            }
            if (cur != null && cur.Genome.Count > 0) newPop.Add(cur);
            if (newPop.Count > 0) innerPop = newPop;
        }
        
        // Тест лучшей особи на одном входе
        public double[] TestBest(double[] input)
        {
            var (success, result) = vm.Execute(bestEver.Genome, input);
            return success ? result : null;
        }
    }
    
    // Главный класс программы
    class Program
    {
        static void Main()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            var settings = new Settings();
            var dataLoader = new DataLoader();
            var engine = new EvolutionEngine();
            
            // Загрузка/создание настроек
            if (File.Exists("settings.txt")) settings.Load("settings.txt");
            else settings.Save("settings.txt");
            
            // Запрос пути к данным
            Console.WriteLine("Введите путь к файлу с данными (формат: вход1 вход2 | выход1 выход2):");
            var dataPath = Console.ReadLine();
            if (!dataLoader.Load(dataPath))
            {
                Console.WriteLine("Ошибка загрузки данных. Нажмите Enter для выхода.");
                Console.ReadLine();
                return;
            }
            
            engine.Initialize(settings, dataLoader);
            bool exit = false;
            
            // Главный цикл меню
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
                
                switch (Console.ReadLine())
                {
                    case "1":
                        Console.Write("Сколько поколений эволюции прогонять? ");
                        if (int.TryParse(Console.ReadLine(), out var gen))
                        {
                            Console.WriteLine($"Эволюция на {gen} поколений...");
                            for (int g = 0; g < gen; g++)
                            {
                                engine.EvolveGeneration();
                                if ((g + 1) % 10 == 0 || g == gen - 1)
                                    Console.WriteLine($"Поколение {g + 1}, ошибка: {engine.GetBest().Fitness:e}");
                            }
                            Console.WriteLine($"Готово. Лучшая ошибка: {engine.GetBest().Fitness:e}");
                        }
                        break;
                    case "2":
                        Console.WriteLine("Тестирование (введите 'выйти' для выхода):");
                        while (true)
                        {
                            Console.Write("Вход: ");
                            var line = Console.ReadLine();
                            if (line?.ToLower() == "выйти") break;
                            var inp = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s, CultureInfo.InvariantCulture)).ToArray();
                            var res = engine.TestBest(inp);
                            Console.WriteLine($"Выход: {string.Join(" ", res?.Select(r => r.ToString("G17", CultureInfo.InvariantCulture)) ?? new[] { "ошибка" })}");
                        }
                        break;
                    case "3":
                        Console.Write("Файл для сохранения особи: ");
                        File.WriteAllText(Console.ReadLine(), engine.ExportIndividual(engine.GetBest()));
                        Console.WriteLine("Сохранено.");
                        break;
                    case "4":
                        Console.Write("Файл для сохранения популяции: ");
                        engine.SavePopulation(Console.ReadLine());
                        Console.WriteLine("Популяция сохранена.");
                        break;
                    case "5":
                        Console.Write("Файл для загрузки популяции: ");
                        engine.LoadPopulation(Console.ReadLine());
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
