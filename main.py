import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image
import io
from datetime import datetime, timedelta

# import YOLO framwork
from ultralytics import YOLO


def get_image(data):
    """
    data: historical price in df format with datetime index
    """

    # plot candlestick chart
    fig, ax = mpf.plot(data, type="candle", style="yahoo", ylabel="", ylabel_lower="", axtitle="", figsize=(6.4, 6.4), returnfig=True)

    # convert chart to image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)

    image = Image.open(buf)

    plt.close(fig)

    return image


def pattern_detect(model, source, confidence=0.5):
    """
    model: the trained object detection model (.pt file)
    source: image inputs
    confidence: object confidence threshold for detection
    """

    # model initialize
    model = YOLO(model)

    # model inference
    results = model.predict(source=source, conf=confidence)

    names = results[0].names

    pred = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
          pred_result = {
            "xyxy": box.xyxy.tolist()[0], # bounding box location as
            "conf": round(box.conf[0], 2), # confidence of class detected
            "class" : names[box.cls[0]] # class name
          }

          pred.append(pred_result)

    im_array = results[0].plot()  # plot a BGR numpy array of results
    im = Image.fromarray(im_array[:, :, ::-1])  # RGB PIL image

    buf = io.BytesIO()
    im.save(buf, format="png")

    image = Image.open(buf)

    if len(pred) == 0:
      return image, "No detections"
    else:
      return image, pred
    


def main_function(symbol, period, timeframe, model, conf=0.5, last_day=False):

    # ML model
    model_path = model

    # get current date
    current_datetime = datetime.now()
    new_date = current_datetime - timedelta(days=period)

    # fetch historical price
    df = yf.download(symbol, start=new_date, end=current_datetime, interval=timeframe)

    # total candle stick charts
    rows = len(df)

    # find the number of pictures from rows
    if timeframe == "1d": # crypto = 30 rows, us stock = 20 rows
        num_subplots = round(rows / 15)# if rows >= 30 else 1 # 15 candle sticks per image

    elif timeframe == "4h":
        num_subplots = round(rows / 42) # 21 candle sticks per image

    elif timeframe == "1h": # crypto = 720 rows, us stock = 140 rows
        num_subplots = round(rows / 72) # 72 candle sticks per image

    else:
        num_subplots = round( rows / 30)

    # pull price data from current date
    for i in reversed(range(num_subplots)):
        start_index = i * round(rows / num_subplots)
        end_index = (i + 1) * round(rows / num_subplots) if i < num_subplots - 1 else rows
        subset_data = df.iloc[start_index:end_index]

        # get image
        image = get_image(subset_data)

        # run detection function
        im_pred, pred = pattern_detect(model=model_path, source=image, confidence=conf)

        # find date difference
        when = current_datetime - subset_data.index[0].to_pydatetime().replace(tzinfo=None)
        days = when.days
        hours, seconds = divmod(when.seconds, 3600)

        # result Initialize
        result = {
            "start_date": subset_data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
            "end_date": subset_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            "timeframe" : timeframe,
            "when" : {"days" : days, "hours" : hours},
            "class": ["No Detection"],
            "conf": ["No Detection"],
            "image": im_pred
        }

        # return the result if pattern found and stop looping
        if pred != "No detections":
            result["class"] = [pred[i]["class"] for i in range(len(pred))]
            result["conf"] = [pred[i]["conf"] for i in range(len(pred))]
            result["image"]= im_pred

            return result

        elif last_day == True:
            return result

    # the result is no pattern found
    return result