# blog-code-snippets

```
Install-Package Microsoft.ML
Console.WriteLine("Fenced code blocks ftw!");

```

```cs
Install-Package Microsoft.ML
dotnet add package Microsoft.ML


public class SentimentData
{
    [LoadColumn(0)] public string SentimentText;
    [LoadColumn(1)] public bool Sentiment;
}

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
}


var context = new MLContext();
var data = context.Data.LoadFromTextFile<SentimentData>("data.csv", separatorChar: ',', hasHeader: true);


var tt = context.Data.TrainTestSplit(data);
var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
    .Append(context.Transforms.Concatenate("Features", "Features"))
    .Append(context.Transforms.NormalizeMinMax("Features"))
    .Append(context.Transforms.CopyColumns("Label", nameof(SentimentData.Sentiment)))
    .Append(context.Transforms.TrainTestSplit().TrainSet);


var trainer = context.BinaryClassification.Trainers.SdcaLogisticRegression();
var trainedPipeline = pipeline.Append(trainer).Fit(tt.Train


var predictions = trainedPipeline.Transform(tt.TestSet);
var metrics = context.BinaryClassification.Evaluate(predictions, "Label", "Score", "PredictedLabel");
Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");


context.Model.Save(trainedPipeline, tt.TrainSet.Schema, "model.zip");


var model = context.Model.Load("model.zip", out var modelSchema);
var predictionEngine = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);


var input = new SentimentData { SentimentText = "This is an excellent product!" };
var result = predictionEngine.Predict(input);
Console.WriteLine($"Sentiment: {(result.Prediction ? "Positive" : "Negative")}");



Console.WriteLine("Fenced code blocks ftw!");
```
