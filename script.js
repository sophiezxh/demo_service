
ETTS_data = [{'text': 'No, I burst the balloon!',
  'sid': '1',
  'emotion': 'Sad',
  'origin_audio_path': './static/demo_audio/origin/0011_001166.wav',
  'vits_label_path': './static/demo_audio/vits_label/0011_001166.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0011_001166.wav'},
 {'text': 'First, issue a reward.',
  'sid': '1',
  'emotion': 'Happy',
  'origin_audio_path': './static/demo_audio/origin/0011_000881.wav',
  'vits_label_path': './static/demo_audio/vits_label/0011_000881.wav',
  'mmtts_vits_path': './static/demo_audio/mmtts_vits/0011_000881.wav'},
   ]


emotionColor = {
    'Neutral': '#a0a0a0',
    'Joy': '#ffbf7f',
    'Sadness': '#8fb2ff',
    'Surprise': '#b266ff',
    'Disgust': '#29a329',
    'Happy': '#ffbf7f',
    'Sad': '#8fb2ff',
    'Angry': '#ff4c4c',

    'Happily': '#ffbf7f',
    'Sadly': '#8fb2ff',
    'Fearful': '#ff7f00',
    'Fearfully': '#ff7f00',
    'Angry': '#ff4c4c',
    'Angrily': '#ff4c4c',
    'Disgusted': '#29a329',
    'Disgustingly': '#29a329',
    'Surprised': '#b266ff',
}

function createETTSTable(data) {
    var table = document.getElementById("ETTS_table").getElementsByTagName("tbody")[0];
    for (var i = 0; i < data.length; i++){
        var row = table.insertRow();
        var textCell = row.insertCell(0);
        var speakerCell = row.insertCell(1);
        var emotionCell = row.insertCell(2);
        var originAudioCell = row.insertCell(3);
        var vitsLabelCell = row.insertCell(4);
        var mmttsVitsCell = row.insertCell(5);

        var text = document.createTextNode(data[i].text);
        textCell.appendChild(text);

        var speaker = document.createTextNode(data[i].sid);
        speakerCell.appendChild(speaker);

        var emotion = document.createTextNode(data[i].emotion);
        emotionCell.appendChild(emotion);
        emotionCell.style.color = emotionColor[data[i].emotion];

        var originAudio = document.createElement("audio");
        originAudio.src = data[i].origin_audio_path;
        originAudio.controls = true;
        originAudioCell.appendChild(originAudio);

        var vitsLabel = document.createElement("audio");
        vitsLabel.src = data[i].vits_label_path;
        vitsLabel.controls = true;
        vitsLabelCell.appendChild(vitsLabel);

        var mmttsVits = document.createElement("audio");
        mmttsVits.src = data[i].mmtts_vits_path;
        mmttsVits.controls = true;
        mmttsVitsCell.appendChild(mmttsVits);


    }
}

window.onload = function(){
    createETTSTable(ETTS_data)
}
